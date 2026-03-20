import os
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from dotenv import load_dotenv
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    logging as hf_logging,
)

load_dotenv()

BATCH_SIZE = 8
SAMPLE_DATA_PATH = Path(__file__).parent / "tests" / "data" / "csv" / "mixed_sample.csv"


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def detect_text_column(df: pd.DataFrame) -> str | None:
    return next((col for col in df.columns if df[col].dtype == "object"), None)


@st.cache_resource
def load_model(device):
    """Load model and tokenizer once via @st.cache_resource in float16."""
    model_path = "siebert/sentiment-roberta-large-english"
    token = os.environ.get("HF_TOKEN")
    hf_logging.set_verbosity_error()
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, dtype=torch.float16, token=token
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    return model, tokenizer


def process_dataframe(df, text_column, model, tokenizer, device):
    """Classify texts in batches; returns a copy with Sentiment and Confidence columns."""
    texts = df[text_column].astype(str).tolist()
    sentiments = [""] * len(texts)
    confidences = [0.0] * len(texts)
    progress_bar = st.progress(0)

    valid = [(i, t) for i, t in enumerate(texts) if t.strip()]

    if not valid:
        progress_bar.progress(1.0)
    else:
        id2label = model.config.id2label
        indices, valid_texts = zip(*valid)
        total = len(valid_texts)

        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)
            inputs = tokenizer(
                list(valid_texts[start:end]),
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            with torch.inference_mode():
                probs = torch.softmax(model(**inputs).logits, dim=-1)
                max_probs, preds = probs.max(dim=-1)

            for idx, pred, conf in zip(
                indices[start:end], preds.tolist(), max_probs.tolist()
            ):
                sentiments[idx] = id2label[pred].lower()
                confidences[idx] = round(conf, 4)

            progress_bar.progress(end / total)

    result = df.copy()
    result["Sentiment"] = sentiments
    result["Confidence"] = confidences
    return result


st.set_page_config(
    page_title="Text Classification Pipeline",
    page_icon=":mag:",
    layout="wide",
)

device = get_device()

with st.spinner(f"Loading model on {device.upper()}..."):
    model, tokenizer = load_model(device)

st.title("Text Classification Pipeline")
st.write("Classify the sentiment of text in your CSV as positive or negative.")
st.caption(f"Powered by SiEBERT (RoBERTa-large) · Running on {device.upper()}")

col_upload, col_sample = st.columns(2)
with col_upload:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
with col_sample:
    st.write("")
    st.write("")
    use_sample = st.button("Try with sample data")

st.caption("Supports CSV files. Your data is processed locally and never stored.")

if use_sample:
    st.session_state["df"] = pd.read_csv(SAMPLE_DATA_PATH)
    st.session_state["source_name"] = "mixed_sample"
elif uploaded_file is not None:
    try:
        st.session_state["df"] = pd.read_csv(uploaded_file)
        st.session_state["source_name"] = uploaded_file.name.rsplit(".", 1)[0]
    except (
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        UnicodeDecodeError,
        ValueError,
    ):
        st.error("Could not read this file. Please check it's a valid CSV.")

df = st.session_state.get("df")
source_name = st.session_state.get("source_name", "")

if df is not None:
    if df.empty:
        st.warning("This CSV has no rows. Please upload a file with data.")
    elif (default_col := detect_text_column(df)) is None:
        st.warning("No text columns detected. Please check your CSV.")
    else:
        st.subheader("Select the column containing text to classify")

        columns = df.columns.tolist()
        text_column = st.selectbox(
            "Text column", options=columns, index=columns.index(default_col)
        )

        st.write("Preview of selected column")
        st.dataframe(df[[text_column]].head(), width="stretch")

        col_classify, col_reset = st.columns([1, 1])
        with col_classify:
            classify_clicked = st.button("Classify", type="primary")
        with col_reset:
            if st.button("Start over"):
                for key in ["df", "source_name"]:
                    st.session_state.pop(key, None)
                st.rerun()

        if classify_clicked:
            with st.spinner("Classifying..."):
                result_df = process_dataframe(df, text_column, model, tokenizer, device)

            csv_data = result_df.to_csv(index=False)

            if result_df["Sentiment"].eq("").all():
                st.info(
                    "All values in this column are empty. "
                    "No classification was performed."
                )
            else:
                st.success("Classification complete!")

                total = len(result_df)
                classified = result_df[result_df["Sentiment"] != ""]
                pos_count = int((classified["Sentiment"] == "positive").sum())
                neg_count = int((classified["Sentiment"] == "negative").sum())
                avg_conf = classified["Confidence"].mean() if len(classified) else 0.0

                # total > 0 guaranteed: df.empty and all-blank branches exit above
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total rows", total)
                m2.metric("Positive", f"{pos_count} ({pos_count / total * 100:.0f}%)")
                m3.metric("Negative", f"{neg_count} ({neg_count / total * 100:.0f}%)")
                m4.metric("Avg confidence", f"{avg_conf:.1%}")

                st.dataframe(result_df, width="stretch")

            st.download_button(
                label="Download results as CSV",
                data=csv_data,
                file_name=f"{source_name}_sentiment.csv",
                mime="text/csv",
            )
