#!/usr/bin/env python3
"""Test client for NeMo streaming ASR WebSocket endpoint."""

import asyncio
import json
import sys
import wave

import websockets


async def stream_audio(ws_url: str, audio_path: str, chunk_ms: int = 100):
    """Stream audio file to WebSocket endpoint."""
    # Read WAV file
    with wave.open(audio_path, "rb") as wav:
        sample_rate = wav.getframerate()
        n_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        audio_data = wav.readframes(wav.getnframes())

    if sample_rate != 16000:
        print(f"Warning: Expected 16kHz, got {sample_rate}Hz")
    if n_channels != 1:
        print(f"Warning: Expected mono, got {n_channels} channels")
    if sample_width != 2:
        print(f"Warning: Expected 16-bit, got {sample_width * 8}-bit")

    chunk_size = int(sample_rate * chunk_ms / 1000) * 2  # bytes per chunk

    async with websockets.connect(ws_url) as ws:
        # Wait for READY
        ready = await ws.recv()
        print(f"Server: {ready}")

        # Send audio chunks
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i : i + chunk_size]
            await ws.send(chunk)
            print(f"Sent {len(chunk)} bytes")

            # Check for responses (non-blocking)
            try:
                while True:
                    response = await asyncio.wait_for(ws.recv(), timeout=0.01)
                    data = json.loads(response)
                    status = "Final" if data.get("is_final") else "Partial"
                    print(f"{status}: {data['text']}")
            except asyncio.TimeoutError:
                pass

            await asyncio.sleep(chunk_ms / 1000)

        # Signal end
        await ws.send("END")

        # Get final response
        while True:
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(response)
                status = "Final" if data.get("is_final") else "Partial"
                print(f"{status}: {data['text']}")
                if data.get("is_final"):
                    break
            except asyncio.TimeoutError:
                break


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <ws_url> <audio.wav>")
        sys.exit(1)

    asyncio.run(stream_audio(sys.argv[1], sys.argv[2]))
