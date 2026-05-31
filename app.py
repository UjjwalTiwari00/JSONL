import json
import os
import shutil
import tempfile

import streamlit as st
import pandas as pd

from extractor import (
    extract_text_from_file,
    build_qa_pairs,
    save_jsonl,
    save_to_mongodb,
    test_mongodb_connection,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="QA Dataset Builder", layout="wide", page_icon="📄")

st.title("📄 QA Dataset Builder")
st.caption(
    "Upload PDFs or DOCX files → extract Q&A pairs → view & download JSONL → push to MongoDB."
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "qa_pairs" not in st.session_state:
    st.session_state.qa_pairs = []
if "mongo_status" not in st.session_state:
    st.session_state.mongo_status = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("🤖 AI Question Generation")
    hf_token = st.text_input(
        "HuggingFace API Token (optional)",
        type="password",
        placeholder="hf_xxxxxxxxxxxxxx",
        help=(
            "Free token from huggingface.co/settings/tokens.\n"
            "Enables smart question generation via the cloud — no download required.\n"
            "Leave blank to use fast rule-based generation instead."
        ),
    )
    if hf_token:
        st.success("AI mode active — questions generated via HuggingFace API.")
    else:
        st.info("Rule-based mode — fast, offline, no API needed.")

    st.divider()

    st.subheader("🗄️ MongoDB")
    # Read from Streamlit secrets if available (set in Streamlit Cloud dashboard)
    _default_uri = st.secrets.get("MONGO_URI", "") if hasattr(st, "secrets") else ""
    mongo_uri = st.text_input(
        "MongoDB URI",
        value=_default_uri,
        type="password",
        placeholder="mongodb+srv://user:pass@cluster.mongodb.net/",
    )
    db_name = st.text_input("Database", value="doc_qa")
    collection_name = st.text_input("Collection", value="qa_pairs")

    if st.button("🔌 Test Connection", use_container_width=True):
        with st.spinner("Connecting…"):
            ok = test_mongodb_connection(mongo_uri)
        if ok:
            st.success("Connected successfully!")
        else:
            st.error("Connection failed. Check your URI.")

    st.divider()
    output_file = st.text_input("JSONL filename", value="output.jsonl")

# ---------------------------------------------------------------------------
# Step 1 — Upload
# ---------------------------------------------------------------------------
st.subheader("Step 1 — Upload Documents")

uploaded_files = st.file_uploader(
    "PDF, DOCX, TXT, MD or RST",
    type=["pdf", "docx", "txt", "md", "rst"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.info(f"{len(uploaded_files)} file(s) ready: {', '.join(f.name for f in uploaded_files)}")

# ---------------------------------------------------------------------------
# Step 2 — Extract
# ---------------------------------------------------------------------------
st.subheader("Step 2 — Extract Q&A Pairs")

col1, col2 = st.columns([1, 3])
with col1:
    run_btn = st.button("▶ Run Extraction", type="primary", use_container_width=True)
with col2:
    if st.session_state.qa_pairs:
        st.success(f"✅ {len(st.session_state.qa_pairs)} Q&A pairs ready")

if run_btn:
    if not uploaded_files:
        st.warning("Upload at least one file first.")
    else:
        all_qas = []
        temp_dirs = []
        file_count = len(uploaded_files)
        progress_bar = st.progress(0, text="Starting…")
        status_text = st.empty()

        for file_idx, uploaded in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded.name} ({file_idx + 1}/{file_count})…")

            temp_dir = tempfile.mkdtemp(prefix="upload_")
            temp_dirs.append(temp_dir)
            temp_path = os.path.join(temp_dir, uploaded.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded.getbuffer())

            text = extract_text_from_file(temp_path)
            if not text.strip():
                st.warning(f"No text extracted from {uploaded.name}, skipping.")
                continue

            base = file_idx / file_count
            share = 1.0 / file_count

            def make_cb(b, s, bar, name):
                def cb(frac):
                    bar.progress(min(b + frac * s, 1.0), text=f"Generating questions for {name}…")
                return cb

            qas = build_qa_pairs(
                text,
                source=uploaded.name,
                hf_token=hf_token or None,
                progress_callback=make_cb(base, share, progress_bar, uploaded.name),
            )
            all_qas.extend(qas)

        for temp_dir in temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)

        progress_bar.progress(1.0, text="Done!")
        status_text.empty()

        if all_qas:
            st.session_state.qa_pairs = all_qas
            st.session_state.mongo_status = None
            st.success(f"Extracted **{len(all_qas)}** Q&A pairs.")
        else:
            st.error("No Q&A pairs could be extracted.")

# ---------------------------------------------------------------------------
# Step 3 — View JSONL
# ---------------------------------------------------------------------------
if st.session_state.qa_pairs:
    qas = st.session_state.qa_pairs

    st.subheader("Step 3 — View & Download JSONL")

    search = st.text_input("🔍 Filter", placeholder="Search questions, answers, or context…")
    filtered = qas
    if search:
        q = search.lower()
        filtered = [
            item for item in qas
            if q in item.get("question", "").lower()
            or q in item.get("answer", "").lower()
            or q in item.get("context", "").lower()
        ]

    st.caption(f"Showing {len(filtered)} of {len(qas)} pairs")

    df = pd.DataFrame(filtered)[["source", "question", "answer", "context"]]
    st.dataframe(df, use_container_width=True, height=420)

    with st.expander("📋 Raw JSONL (first 20 lines)"):
        st.code(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in filtered[:20]),
            language="json",
        )

    fname = output_file if output_file.endswith(".jsonl") else output_file + ".jsonl"
    st.download_button(
        label="⬇️ Download JSONL",
        data="\n".join(json.dumps(item, ensure_ascii=False) for item in qas),
        file_name=fname,
        mime="text/plain",
        use_container_width=True,
    )

    # ---------------------------------------------------------------------------
    # Step 4 — Push to MongoDB
    # ---------------------------------------------------------------------------
    st.subheader("Step 4 — Push to MongoDB")

    if not mongo_uri:
        st.info("Enter your MongoDB URI in the sidebar.")
    else:
        col_a, col_b = st.columns([1, 3])
        with col_a:
            push_btn = st.button("🚀 Push to MongoDB", type="primary", use_container_width=True)
        with col_b:
            if st.session_state.mongo_status == "success":
                st.success(f"✅ Inserted into `{db_name}.{collection_name}`")
            elif st.session_state.mongo_status == "error":
                st.error("Push failed — check the connection test in the sidebar.")

        if push_btn:
            with st.spinner(f"Inserting {len(qas)} records…"):
                try:
                    result = save_to_mongodb(mongo_uri, db_name, collection_name, qas)
                    st.session_state.mongo_status = "success"
                    st.success(
                        f"✅ Inserted **{result['inserted_count']}** records into "
                        f"`{db_name}.{collection_name}`"
                    )
                except Exception as e:
                    st.session_state.mongo_status = "error"
                    st.error(f"MongoDB error: {e}")
