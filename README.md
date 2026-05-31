# Document QA Extractor

A Python Streamlit app to extract text from PDF, DOCX, TXT, Markdown, and GitHub repos, then convert content into question-answer JSONL records.

## Features

- Extract text from PDF/DOCX/TXT/Markdown files.
- Scan local folders and GitHub repositories.
- Build QA pairs using document structure and definitions only (no LLM usage).
- Save output to a JSONL file.
- Insert records into MongoDB.

## Setup

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run app.py
```

## Usage

1. Upload documents or provide a local folder path.
2. Optionally provide a GitHub repository URL.
3. Click `Run extraction`.
4. Download the JSONL file or insert directly into MongoDB.

## CLI Usage

This repository also includes a command-line extraction tool:

```bash
python extractor_cli.py --files doc1.pdf doc2.docx --folder path/to/folder --repo https://github.com/user/repo --jsonl output.jsonl --mongo-uri "mongodb+srv://username:password@cluster0.mongodb.net" --mongo-db doc_qa --mongo-collection qa_pairs --include-code
```

Use the generated JSONL file for later LLM training or dataset preparation.

## Hosting

- Deploy on Streamlit Cloud by linking this repository.
- Or host on any server with Python and open port for Streamlit.

## MongoDB

Provide your MongoDB connection URI in the sidebar, for example:

```text
mongodb+srv://username:password@cluster0.mongodb.net
```

Then click the `Save` button to store QA records in MongoDB.

## Notes

- This project avoids LLMs and instead uses heuristic QA generation from headings, definitions, and text chunks.
- For higher-quality question generation, a later upgrade can add semantic embeddings or model-based QA.
