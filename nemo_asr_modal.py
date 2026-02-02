import modal

app = modal.App("nemotron-asr-streaming")

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
    .copy_local_file("speech_to_text_cache_aware_streaming_infer.py", "/root/speech_to_text_cache_aware_streaming_infer.py")
)

hf_model = modal.Volume.from_name("hf-model-cache", create_if_missing=True)


@app.cls(
    image=image,
    gpu=modal.gpu.H100(count=1),
    timeout=600,
    container_idle_timeout=300,
    allow_concurrent_inputs=10,
    volumes={"/model-cache": hf_model},
)
class NeMoASR:
    hf_repo: str = "TrySalient/nemotron-asr-5k-combined-epoch15"
    model_filename: str = "nemotron-asr-5k-combined-epoch15.nemo"
    att_context_size: list = [70, 1]

    @modal.enter()
    def setup(self):
        import torch
        import nemo.collections.asr as nemo_asr
        from huggingface_hub import hf_hub_download
        import os
        import sys

        # Add the script to path
        sys.path.insert(0, "/root")

        # Import functions from the exact NeMo script
        from speech_to_text_cache_aware_streaming_infer import (
            perform_streaming,
            extract_transcriptions,
            calc_drop_extra_pre_encoded,
        )
        self.perform_streaming = perform_streaming
        self.extract_transcriptions = extract_transcriptions

        from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
        self.CacheAwareStreamingAudioBuffer = CacheAwareStreamingAudioBuffer

        torch.set_grad_enabled(False)
        self.device = torch.device("cuda")

        # Download model
        cache_dir = "/model-cache"
        model_path = os.path.join(cache_dir, self.model_filename)

        if not os.path.exists(model_path):
            print(f"Downloading {self.hf_repo}...")
            model_path = hf_hub_download(
                repo_id=self.hf_repo,
                filename=self.model_filename,
                cache_dir=cache_dir,
                local_dir=cache_dir,
            )
            hf_model.commit()

        print(f"Loading model from {model_path}...")
        self.asr_model = nemo_asr.models.ASRModel.restore_from(
            restore_path=model_path,
            map_location=self.device,
        )

        # Set att_context_size
        if hasattr(self.asr_model.encoder, "set_default_att_context_size"):
            self.asr_model.encoder.set_default_att_context_size(att_context_size=self.att_context_size)
            print(f"Set att_context_size to {self.att_context_size}")

        # Disable CUDA graphs for streaming
        if hasattr(self.asr_model, 'decoding') and hasattr(self.asr_model.decoding, 'rnnt_decoding_config'):
            self.asr_model.decoding.rnnt_decoding_config.cuda_graph_mode = None

        self.asr_model = self.asr_model.to(device=self.device, dtype=torch.float32)
        self.asr_model.eval()

        self._amp_dtype = torch.float16
        self._compute_dtype = torch.float32

        print(f"Model loaded. Streaming config: {self.asr_model.encoder.streaming_cfg}")

    @modal.web_endpoint(method="GET")
    def health(self):
        return {"status": "healthy", "model": self.hf_repo}

    @modal.asgi_app()
    def webapp(self):
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
        from fastapi.responses import JSONResponse
        import torch
        import tempfile
        import os
        import numpy as np
        import soundfile as sf

        web_app = FastAPI()

        @web_app.post("/transcribe")
        async def transcribe_upload(file: UploadFile = File(...)):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                content = await file.read()
                f.write(content)
                audio_path = f.name

            try:
                streaming_buffer = self.CacheAwareStreamingAudioBuffer(
                    model=self.asr_model,
                    online_normalization=False,
                    pad_and_drop_preencoded=False,
                )
                _ = streaming_buffer.append_audio_file(audio_path, stream_id=-1)

                with torch.amp.autocast('cuda', dtype=self._amp_dtype, enabled=True):
                    final_streaming_tran, _ = self.perform_streaming(
                        asr_model=self.asr_model,
                        streaming_buffer=streaming_buffer,
                        compute_dtype=self._compute_dtype,
                        compare_vs_offline=False,
                        debug_mode=False,
                        pad_and_drop_preencoded=False,
                    )

                return JSONResponse({"text": final_streaming_tran[0] if final_streaming_tran else ""})
            finally:
                os.unlink(audio_path)

        @web_app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            await websocket.send_text("READY")

            audio_chunks = []

            try:
                while True:
                    message = await websocket.receive()

                    if "text" in message and message["text"] == "END":
                        if audio_chunks:
                            audio_np = np.concatenate(audio_chunks)

                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                                sf.write(f.name, audio_np, 16000)
                                audio_path = f.name

                            try:
                                streaming_buffer = self.CacheAwareStreamingAudioBuffer(
                                    model=self.asr_model,
                                    online_normalization=False,
                                    pad_and_drop_preencoded=False,
                                )
                                _ = streaming_buffer.append_audio_file(audio_path, stream_id=-1)

                                with torch.amp.autocast('cuda', dtype=self._amp_dtype, enabled=True):
                                    final_streaming_tran, _ = self.perform_streaming(
                                        asr_model=self.asr_model,
                                        streaming_buffer=streaming_buffer,
                                        compute_dtype=self._compute_dtype,
                                        compare_vs_offline=False,
                                        debug_mode=False,
                                        pad_and_drop_preencoded=False,
                                    )

                                await websocket.send_json({
                                    "text": final_streaming_tran[0] if final_streaming_tran else "",
                                    "is_final": True
                                })
                            finally:
                                os.unlink(audio_path)
                        break

                    elif "bytes" in message:
                        audio_chunk = np.frombuffer(message["bytes"], dtype=np.int16).astype(np.float32) / 32768.0
                        audio_chunks.append(audio_chunk)

            except WebSocketDisconnect:
                pass

        return web_app


@app.local_entrypoint()
def main():
    print("Deploy with: modal deploy nemo_asr_modal.py")
