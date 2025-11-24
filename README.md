# Subtitle Translator

A small Python utility to translate `.srt` subtitle files using an LLM. It supports both cloud APIs (DeepSeek) and local LLMs (Ollama, LM Studio).

This project reads SRT files from the `input/` folder, translates them in parallel using an LLM, and writes translated `.srt` files to the `output/` folder.

Key features
- Preserves SRT structure (index, timestamps) and simple HTML tags like `<i>` and `<b>`.
- Splits subtitles into chunks and translates them in parallel for speed.
- Skips already-translated files in `output/` by default.
- **Dual Mode:** Use `llm_translate.py` for high-speed cloud API translation or `local_translate.py` for free, private local translation.

Quick Start

1. Python and virtualenv

Make sure you have Python 3.11+ installed. In the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Configure environment

Copy the example env and fill in your details:

```bash
cp .env.example .env
```

- **For DeepSeek API:** Set `DEEPSEEK_API_KEY`.
- **For Local LLM:** Set `OLLAMA_BASE_URL` (e.g., `http://localhost:11434/v1` for Ollama or `http://localhost:1234/v1` for LM Studio) and `OLLAMA_MODEL`.

4. Prepare input files

Place `.srt` files into the `input/` folder. Files in `temp/` are ignored by default.

5. Run the translator

**Option A: Use DeepSeek API (Fastest)**
```bash
python llm_translate.py
```
*Uses high concurrency (50 workers) for maximum speed.*

**Option B: Use Local LLM (Free/Private)**
```bash
python local_translate.py
```
*Uses 1 worker to ensure stability and avoid overloading your local hardware. Requires Ollama or LM Studio running.*

Outputs will be written to `output/` with the pattern: `original_filename.<TARGET_LANGUAGE>.srt`. For example: `episode1.pt-BR.srt`.

Configuration and tuning

- `TARGET_LANGUAGE` is read from `.env`. Example: `pt-BR`, `es-ES`.
- `OLLAMA_MODEL`: The model name to use locally (e.g., `deepseek-r1`, `llama3.2`).
- `OLLAMA_BASE_URL`: The API endpoint for your local server.
- Concurrency and chunking are controlled in `utils.py` and the respective scripts:
  - `chunk_blocks(blocks, batch_size=...)` — change `batch_size` to make chunks smaller/larger.
  - `llm_translate.py` defaults to 50 workers.
  - `local_translate.py` defaults to 1 worker.

Notes & Best Practices

- **Local Inference:** For best performance with local models, use **LM Studio** with GPU offloading enabled. Ollama is also supported but may require manual configuration for GPU support on some systems (e.g., AMD ROCm).
- Do NOT commit your `.env` file. Use `.env.example` and keep `.env` private.
- The script will overwrite any existing file with the same output name unless it detects and skips existing translations (it currently skips files already present in `output/`).
- If you encounter API rate limits (HTTP 429) with DeepSeek, reduce `max_workers` in `llm_translate.py`.

Troubleshooting

- **Local Translation Timeout:** If `local_translate.py` times out, ensure your local model is loaded and responsive. You can test a single chunk with:
  ```bash
  python test_local_chunk.py --provider ollama
  ```
- **Validation Errors:** If the output format is wrong, try a more capable model (e.g., `llama3.1:8b` or `mistral`) or check if the model is outputting markdown artifacts (the script attempts to strip them automatically).

License & Security

- Keep your `DEEPSEEK_API_KEY` private. Never commit `.env` to version control.

