import os
from pathlib import Path

import mlx.core as mx
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from mlx_transformers.models import RobertaForSequenceClassification
from transformers import (
    AutoConfig,
    AutoTokenizer,
    logging as hf_logging,
)

load_dotenv()

BATCH_SIZE = 8
SAMPLE_DATA_PATH = Path(__file__).parent / "tests" / "data" / "csv" / "mixed_sample.csv"


def detect_text_column(df: pd.DataFrame) -> str | None:
    return next((col for col in df.columns if df[col].dtype == "object"), None)


@st.cache_resource
def load_model():
    """Load model and tokenizer once via @st.cache_resource in float16."""
    model_path = "siebert/sentiment-roberta-large-english"
    token = os.environ.get("HF_TOKEN")
    hf_logging.set_verbosity_error()
    config = AutoConfig.from_pretrained(model_path, token=token)
    model = RobertaForSequenceClassification(config)
    model.from_pretrained(model_path, float16=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    return model, tokenizer


def process_dataframe(df, text_column, model, tokenizer):
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
                return_tensors="np",
                padding=True,
                truncation=True,
            )
            inputs = {k: mx.array(v) for k, v in inputs.items()}

            probs = mx.softmax(model(**inputs).logits, axis=-1)
            max_probs = mx.max(probs, axis=-1)
            preds = mx.argmax(probs, axis=-1)
            mx.eval(max_probs, preds)

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

with st.spinner("Loading model..."):
    model, tokenizer = load_model()

st.title("Text Classification Pipeline")
st.write("Classify the sentiment of text in your CSV as positive or negative.")
st.caption("Powered by SiEBERT (RoBERTa-large) · Running on MLX (Apple Silicon)")

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
                result_df = process_dataframe(df, text_column, model, tokenizer)

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
