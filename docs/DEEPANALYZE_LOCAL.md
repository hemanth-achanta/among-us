# DeepAnalyze Local Setup (No API)

This app can generate **data science reports** using [DeepAnalyze](https://github.com/ruc-datalab/DeepAnalyze) running **locally** on your machine. No external API and no DeepAnalyze API server (port 8200) are used. All traffic stays between the app and a local vLLM server.

## How it works

- **vLLM** runs on your machine and serves the DeepAnalyze-8B model. It exposes an OpenAI-compatible HTTP API (e.g. `http://localhost:8000/v1`).
- The **app** sends report-generation requests to this local URL with the data in the prompt. No file upload to any external service.

## 1. Environment

- **Python**: 3.12+ recommended (for DeepAnalyze/vLLM).
- **GPU**: Recommended; see [DeepAnalyze README](https://github.com/ruc-datalab/DeepAnalyze) for memory vs quantized models. CPU is possible but slow.
- Optional: `conda create -n deepanalyze python=3.12 -y`.

## 2. Download the model

- From **Hugging Face**: [RUC-DataLab/DeepAnalyze-8B](https://huggingface.co/RUC-DataLab/DeepAnalyze-8B).
- Or use the model ID directly with vLLM (see below). vLLM downloads it on first `vllm serve` into the Hugging Face cache.

**Cache location:** `~/.cache/huggingface/hub/models--RUC-DataLab--DeepAnalyze-8B/` (override with `HF_HOME` or `HUGGINGFACE_HUB_CACHE`).

**Track download size (run anytime):**
```bash
du -sh ~/.cache/huggingface/hub/models--RUC-DataLab--DeepAnalyze-8B
```
**Watch size every 5 seconds until you stop (Ctrl+C):**
```bash
# Linux (if watch is installed)
watch -n 5 'du -sh ~/.cache/huggingface/hub/models--RUC-DataLab--DeepAnalyze-8B'

# macOS (no watch): run in a loop
while true; do du -sh ~/.cache/huggingface/hub/models--RUC-DataLab--DeepAnalyze-8B 2>/dev/null; sleep 5; done
```

## 3. Install and run vLLM (local server only)

Install vLLM (in a separate env or the same one):

```bash
pip install vllm>=0.8.5
```

Start the **local** server (no external API):

```bash
vllm serve RUC-DataLab/DeepAnalyze-8B --port 8000
```

Or with a local path:

```bash
vllm serve /path/to/DeepAnalyze-8B --port 8000
```

**Limited memory (CPU or 8 GiB KV cache):** If you see `KV cache memory (8.0 GiB)` / "max seq len (131072)" errors, reduce the context length so the KV cache fits:

```bash
vllm serve RUC-DataLab/DeepAnalyze-8B --port 8000 --max-model-len 8192
```

For 8 GiB KV cache, vLLM may suggest `--max-model-len 58240`; `8192` is safer and enough for typical report prompts. For limited GPU memory, use a quantized model and flags from the DeepAnalyze README (e.g. `--kv-cache-dtype fp8`, `--max-model-len 8192`).

The server listens on `http://localhost:8000/v1` (OpenAI-compatible). No DeepAnalyze API server (8200) is used.

## 4. Verify vLLM

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"RUC-DataLab/DeepAnalyze-8B","messages":[{"role":"user","content":"Hello"}],"max_tokens":100}'
```

vLLM serves under the Hugging Face repo ID; use that exact name in requests. To see the name your server uses: `curl -s http://localhost:8000/v1/models`.

## 5. Enable in the app

Set in your `.env` (or environment):

- `DEEPANALYZE_ENABLED=true`
- `DEEPANALYZE_BASE_URL=http://localhost:8000/v1` (default)
- Optional: `DEEPANALYZE_MODEL=RUC-DataLab/DeepAnalyze-8B` (default), `DEEPANALYZE_REQUEST_TIMEOUT=120`

Then run the Streamlit app. After you run a question and get results, the **Generate report with DeepAnalyze** button appears. You can also use **Report from uploaded file** (CSV or Excel) in the expander.

## 6. Running vLLM in the background

- **Terminal**: Run `vllm serve ...` in a separate terminal or use `nohup` / `tmux`.
- **Process manager**: Run vLLM as a service (e.g. systemd, Docker) so it stays up.

## Summary

| Component                     | Role |
|------------------------------|------|
| **vLLM**                     | Runs on your machine, serves DeepAnalyze-8B, exposes `http://localhost:8000/v1`. |
| **App**                      | Sends report requests to `localhost:8000` only; data is in the prompt. |
| **DeepAnalyze API server (8200)** | Not used. |
| **External/cloud API**       | Not used. |
