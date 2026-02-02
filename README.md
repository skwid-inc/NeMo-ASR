# NeMo FastConformer Streaming ASR

Cache-aware streaming ASR on Modal with H100.

## Deploy

```bash
modal deploy nemo_asr_modal.py
```

## WebSocket Protocol

1. Connect to `wss://<modal-url>/ws`
2. Wait for `READY`
3. Send audio chunks (16kHz, 16-bit PCM)
4. Receive JSON: `{"text": "...", "is_final": false}`
5. Send `END` when done
6. Receive final response with `is_final: true`
