import modal

app = modal.App("nemo-fastconformer-streaming")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "numpy<2",
        "omegaconf",
        "hydra-core",
        "lightning",
        "webdataset",
        "lhotse>=1.22.0",
        "soundfile",
        "librosa",
        "scipy",
        "editdistance",
        "jiwer",
        "pandas",
        "huggingface_hub",
        "sentencepiece",
        "youtokentome",
        "fastapi",
        "uvicorn",
        "websockets",
    )
    .pip_install("nemo_toolkit[asr]==2.2.1")
)


@app.cls(
    image=image,
    gpu=modal.gpu.H100(count=1),
    timeout=600,
    container_idle_timeout=300,
    allow_concurrent_inputs=10,
)
class NeMoASR:
    model_name: str = "nvidia/parakeet-tdt-0.6b-v2"

    @modal.enter()
    def load_model(self):
        import torch
        import nemo.collections.asr as nemo_asr
        from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

        self.device = torch.device("cuda")

        # Load the streaming ASR model
        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=self.model_name,
            map_location=self.device,
        )
        self.asr_model.eval()
        self.asr_model = self.asr_model.to(self.device)

        # Disable CUDA graphs for streaming compatibility
        if hasattr(self.asr_model, 'decoding') and hasattr(self.asr_model.decoding, 'rnnt_decoding_config'):
            self.asr_model.decoding.rnnt_decoding_config.cuda_graph_mode = None

        self.streaming_buffer_class = CacheAwareStreamingAudioBuffer
        print(f"Model {self.model_name} loaded successfully")

    def _create_streaming_buffer(self):
        return self.streaming_buffer_class(
            model=self.asr_model,
            online_normalization=False,
            pad_and_drop_preencoded=False,
        )

    def _perform_streaming_step(self, streaming_buffer, cache_state, previous_hypotheses, pred_out_stream, step_num):
        import torch

        cache_last_channel, cache_last_time, cache_last_channel_len = cache_state

        chunk_audio, chunk_lengths = next(iter(streaming_buffer))
        chunk_audio = chunk_audio.to(self.device)

        # Calculate drop_extra_pre_encoded
        if step_num == 0:
            drop_extra = 0
        else:
            drop_extra = self.asr_model.encoder.streaming_cfg.drop_extra_pre_encoded

        with torch.inference_mode():
            (
                pred_out_stream,
                transcribed_texts,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
                previous_hypotheses,
            ) = self.asr_model.conformer_stream_step(
                processed_signal=chunk_audio,
                processed_signal_length=chunk_lengths,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
                keep_all_outputs=streaming_buffer.is_buffer_empty(),
                previous_hypotheses=previous_hypotheses,
                previous_pred_out=pred_out_stream,
                drop_extra_pre_encoded=drop_extra,
                return_transcription=True,
            )

        # Extract text from hypothesis
        from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
        if isinstance(transcribed_texts[0], Hypothesis):
            text = transcribed_texts[0].text
        else:
            text = transcribed_texts[0]

        new_cache_state = (cache_last_channel, cache_last_time, cache_last_channel_len)
        return text, new_cache_state, previous_hypotheses, pred_out_stream

    @modal.web_endpoint(method="GET")
    def health(self):
        return {"status": "healthy", "model": self.model_name}

    @modal.asgi_app()
    def webapp(self):
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        import numpy as np
        import torch

        web_app = FastAPI()

        @web_app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            await websocket.send_text("READY")

            streaming_buffer = self._create_streaming_buffer()
            cache_state = self.asr_model.encoder.get_initial_cache_state(batch_size=1)
            previous_hypotheses = None
            pred_out_stream = None
            step_num = 0

            audio_buffer = []
            sample_rate = 16000
            # Chunk size in samples (matching model's expected chunk)
            chunk_samples = int(0.64 * sample_rate)  # 640ms chunks

            try:
                while True:
                    message = await websocket.receive()

                    if "text" in message:
                        if message["text"] == "END":
                            # Process remaining audio
                            if audio_buffer:
                                audio_np = np.concatenate(audio_buffer)
                                audio_tensor = torch.from_numpy(audio_np).float().unsqueeze(0)
                                streaming_buffer.append_audio(audio_tensor, torch.tensor([len(audio_np)]))

                                while not streaming_buffer.is_buffer_empty():
                                    text, cache_state, previous_hypotheses, pred_out_stream = self._perform_streaming_step(
                                        streaming_buffer, cache_state, previous_hypotheses, pred_out_stream, step_num
                                    )
                                    step_num += 1

                                await websocket.send_json({
                                    "text": text,
                                    "is_final": True
                                })
                            break

                    elif "bytes" in message:
                        # Receive raw PCM audio (16-bit signed int, 16kHz)
                        audio_bytes = message["bytes"]
                        audio_chunk = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                        audio_buffer.append(audio_chunk)

                        # Process when we have enough samples
                        total_samples = sum(len(c) for c in audio_buffer)
                        if total_samples >= chunk_samples:
                            audio_np = np.concatenate(audio_buffer)
                            # Keep remainder for next chunk
                            process_samples = (total_samples // chunk_samples) * chunk_samples
                            to_process = audio_np[:process_samples]
                            audio_buffer = [audio_np[process_samples:]] if process_samples < len(audio_np) else []

                            audio_tensor = torch.from_numpy(to_process).float().unsqueeze(0)
                            streaming_buffer.append_audio(audio_tensor, torch.tensor([len(to_process)]))

                            # Process all available chunks
                            while not streaming_buffer.is_buffer_empty():
                                text, cache_state, previous_hypotheses, pred_out_stream = self._perform_streaming_step(
                                    streaming_buffer, cache_state, previous_hypotheses, pred_out_stream, step_num
                                )
                                step_num += 1

                                await websocket.send_json({
                                    "text": text,
                                    "is_final": False
                                })

            except WebSocketDisconnect:
                pass

        return web_app


@app.local_entrypoint()
def main():
    print("Deploy with: modal deploy nemo_asr_modal.py")
