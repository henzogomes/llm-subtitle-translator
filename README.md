# Subtitle Translator

A small Python utility to translate `.srt` subtitle files using a DeepSeek (OpenAI-compatible) LLM.

This project reads SRT files from the `input/` folder, translates them in parallel using an LLM, and writes translated `.srt` files to the `output/` folder.

Key features
- Preserves SRT structure (index, timestamps) and simple HTML tags like `<i>` and `<b>`.
- Splits subtitles into chunks and translates them in parallel for speed.
- Skips already-translated files in `output/` by default.

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

Copy the example env and fill in your DeepSeek API key:

```bash
cp .env.example .env
# then edit .env and replace the placeholder API key with your real key
```

4. Prepare input files

Place `.srt` files into the `input/` folder. Files in `temp/` are ignored by default.

5. Run the translator

```bash
python translate.py
```

Outputs will be written to `output/` with the pattern: `original_filename.<TARGET_LANGUAGE>.srt`. For example: `episode1.pt-BR.srt`.

Configuration and tuning

- `TARGET_LANGUAGE` is read from `.env`. Example: `pt-BR`, `es-ES`.
- Concurrency and chunking are controlled in `translate.py`:
  - `chunk_blocks(blocks, batch_size=...)` — change `batch_size` to make chunks smaller/larger.
  - The global executor uses a `max_workers` value; by default the script dynamically sets this to a high value to maximize throughput. You can lower it if you run into rate limits.

Notes & Best Practices

- Do NOT commit your `.env` file. Use `.env.example` and keep `.env` private.
- The script will overwrite any existing file with the same output name unless it detects and skips existing translations (it currently skips files already present in `output/`).
- If you encounter API rate limits (HTTP 429) reduce concurrency (`max_workers`) or increase `batch_size` to reduce request rate.

Troubleshooting

- If translations are slow: try increasing `batch_size` and/or decreasing `max_workers` to find the best balance for speed vs. reliability.
- If you see partial output files or missing sections, check logs for failed batches and re-run — the script resumes by skipping already-completed output files and will process only missing inputs.

License & Security

- Keep your `DEEPSEEK_API_KEY` private. Never commit `.env` to version control.

