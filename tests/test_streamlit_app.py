from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch

from streamlit_app import (
    BATCH_SIZE,
    SAMPLE_DATA_PATH,
    detect_text_column,
    get_device,
    load_model,
    process_dataframe,
)


# --- BATCH_SIZE ---


def test_batch_size_is_positive_int():
    assert isinstance(BATCH_SIZE, int)
    assert BATCH_SIZE > 0


# --- SAMPLE_DATA_PATH ---


def test_sample_data_path_exists():
    assert SAMPLE_DATA_PATH.exists()
    assert SAMPLE_DATA_PATH.suffix == ".csv"


# --- get_device ---


class TestGetDevice:
    @patch("streamlit_app.torch")
    def test_prefers_mps(self, mock_torch):
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.cuda.is_available.return_value = True
        assert get_device() == "mps"

    @patch("streamlit_app.torch")
    def test_falls_back_to_cuda(self, mock_torch):
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = True
        assert get_device() == "cuda"

    @patch("streamlit_app.torch")
    def test_falls_back_to_cpu(self, mock_torch):
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False
        assert get_device() == "cpu"


# --- detect_text_column ---


class TestDetectTextColumn:
    def test_returns_first_object_column(self):
        df = pd.DataFrame({"id": [1, 2], "review": ["good", "bad"], "score": [5, 1]})
        assert detect_text_column(df) == "review"

    def test_skips_non_object_columns(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        assert detect_text_column(df) is None

    def test_returns_first_when_multiple_object_columns(self):
        df = pd.DataFrame({"name": ["Alice", "Bob"], "text": ["hi", "bye"]})
        assert detect_text_column(df) == "name"

    def test_returns_none_for_empty_dataframe(self):
        assert detect_text_column(pd.DataFrame()) is None


# --- load_model ---


class TestLoadModel:
    @patch.dict("os.environ", {"HF_TOKEN": "test-token"})
    @patch("streamlit_app.AutoTokenizer")
    @patch("streamlit_app.AutoModelForSequenceClassification")
    def test_loads_correct_model(self, mock_model_cls, mock_tok_cls):
        mock_tok_cls.from_pretrained.return_value = MagicMock()
        load_model.clear()
        load_model("cpu")

        mock_model_cls.from_pretrained.assert_called_once_with(
            "siebert/sentiment-roberta-large-english",
            dtype=torch.float16,
            token="test-token",
        )
        mock_tok_cls.from_pretrained.assert_called_once_with(
            "siebert/sentiment-roberta-large-english", token="test-token"
        )

    @patch.dict("os.environ", {}, clear=True)
    @patch("streamlit_app.AutoTokenizer")
    @patch("streamlit_app.AutoModelForSequenceClassification")
    def test_loads_without_token(self, mock_model_cls, mock_tok_cls):
        mock_tok_cls.from_pretrained.return_value = MagicMock()
        load_model.clear()
        load_model("cpu")

        mock_model_cls.from_pretrained.assert_called_once_with(
            "siebert/sentiment-roberta-large-english",
            dtype=torch.float16,
            token=None,
        )
        mock_tok_cls.from_pretrained.assert_called_once_with(
            "siebert/sentiment-roberta-large-english", token=None
        )

    @patch("streamlit_app.hf_logging")
    @patch("streamlit_app.AutoTokenizer")
    @patch("streamlit_app.AutoModelForSequenceClassification")
    def test_suppresses_hf_warnings(
        self, mock_model_cls, mock_tok_cls, mock_hf_logging
    ):
        mock_tok_cls.from_pretrained.return_value = MagicMock()
        load_model.clear()
        load_model("cpu")
        mock_hf_logging.set_verbosity_error.assert_called_once()

    @patch("streamlit_app.AutoTokenizer")
    @patch("streamlit_app.AutoModelForSequenceClassification")
    def test_returns_model_and_tokenizer(self, mock_model_cls, mock_tok_cls):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model_cls.from_pretrained.return_value.to.return_value = mock_model
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer

        load_model.clear()
        model, tokenizer = load_model("cpu")

        assert model is mock_model
        assert tokenizer is mock_tokenizer


# --- process_dataframe ---


def _make_mock_tokenizer():
    """Create a mock tokenizer whose inputs support .to(device)."""
    tokenizer = MagicMock()
    mock_inputs = MagicMock()
    mock_inputs.to.return_value = mock_inputs
    tokenizer.return_value = mock_inputs
    return tokenizer


def _make_mock_model(sentiments):
    """Create a mock model returning logits for the given sentiment strings."""
    model = MagicMock()
    model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}

    logits = [[0.0, 1.0] if s == "positive" else [1.0, 0.0] for s in sentiments]

    mock_output = MagicMock()
    mock_output.logits = torch.tensor(logits)
    model.return_value = mock_output
    return model


class TestProcessDataframe:
    @pytest.fixture(autouse=True)
    def _mock_st(self):
        with patch("streamlit_app.st") as mock_st:
            self.mock_progress = MagicMock()
            mock_st.progress.return_value = self.mock_progress
            yield

    def test_adds_sentiment_column(self):
        df = pd.DataFrame({"text": ["good product", "bad product"]})
        result = process_dataframe(
            df,
            "text",
            _make_mock_model(["positive", "negative"]),
            _make_mock_tokenizer(),
            "cpu",
        )
        assert "Sentiment" in result.columns
        assert len(result) == 2

    def test_classifies_positive(self):
        df = pd.DataFrame({"text": ["great"]})
        result = process_dataframe(
            df,
            "text",
            _make_mock_model(["positive"]),
            _make_mock_tokenizer(),
            "cpu",
        )
        assert result["Sentiment"].iloc[0] == "positive"

    def test_classifies_negative(self):
        df = pd.DataFrame({"text": ["terrible"]})
        result = process_dataframe(
            df,
            "text",
            _make_mock_model(["negative"]),
            _make_mock_tokenizer(),
            "cpu",
        )
        assert result["Sentiment"].iloc[0] == "negative"

    def test_maps_labels_to_lowercase(self):
        df = pd.DataFrame({"text": ["great", "awful"]})
        result = process_dataframe(
            df,
            "text",
            _make_mock_model(["positive", "negative"]),
            _make_mock_tokenizer(),
            "cpu",
        )
        assert result["Sentiment"].iloc[0] == "positive"
        assert result["Sentiment"].iloc[1] == "negative"

    def test_batching_multiple_batches(self):
        n = BATCH_SIZE + 3
        df = pd.DataFrame({"text": [f"review {i}" for i in range(n)]})

        model = MagicMock()
        model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}

        batch1_output = MagicMock()
        batch1_output.logits = torch.tensor([[0.0, 1.0]] * BATCH_SIZE)
        batch2_output = MagicMock()
        batch2_output.logits = torch.tensor([[0.0, 1.0]] * 3)
        model.side_effect = [batch1_output, batch2_output]

        result = process_dataframe(df, "text", model, _make_mock_tokenizer(), "cpu")

        assert len(result) == n
        assert model.call_count == 2

    def test_progress_bar_reaches_completion(self):
        df = pd.DataFrame({"text": ["review"]})
        process_dataframe(
            df,
            "text",
            _make_mock_model(["positive"]),
            _make_mock_tokenizer(),
            "cpu",
        )
        last_call_arg = self.mock_progress.progress.call_args_list[-1][0][0]
        assert last_call_arg == pytest.approx(1.0)

    def test_uses_correct_text_column(self):
        df = pd.DataFrame({"col_a": ["ignore"], "col_b": ["use this"]})
        tokenizer = _make_mock_tokenizer()
        process_dataframe(df, "col_b", _make_mock_model(["positive"]), tokenizer, "cpu")
        assert "use this" in tokenizer.call_args[0][0]

    def test_uses_inference_mode(self):
        df = pd.DataFrame({"text": ["test"]})
        with patch("streamlit_app.torch.inference_mode") as mock_inf:
            mock_inf.return_value.__enter__ = MagicMock()
            mock_inf.return_value.__exit__ = MagicMock(return_value=False)
            process_dataframe(
                df,
                "text",
                _make_mock_model(["positive"]),
                _make_mock_tokenizer(),
                "cpu",
            )
            mock_inf.assert_called_once()

    def test_does_not_mutate_input_dataframe(self):
        df = pd.DataFrame({"text": ["review"]})
        result = process_dataframe(
            df,
            "text",
            _make_mock_model(["positive"]),
            _make_mock_tokenizer(),
            "cpu",
        )
        assert "Sentiment" not in df.columns
        assert "Sentiment" in result.columns

    def test_handles_empty_dataframe(self):
        df = pd.DataFrame({"text": []})
        model = MagicMock()
        result = process_dataframe(df, "text", model, MagicMock(), "cpu")

        assert "Sentiment" in result.columns
        assert "Confidence" in result.columns
        assert len(result) == 0
        model.assert_not_called()
        self.mock_progress.progress.assert_called_once_with(1.0)

    def test_tokenizer_called_with_truncation(self):
        df = pd.DataFrame({"text": ["a review"]})
        tokenizer = _make_mock_tokenizer()
        process_dataframe(df, "text", _make_mock_model(["positive"]), tokenizer, "cpu")
        assert tokenizer.call_args[1]["truncation"] is True

    def test_tokenizer_called_with_padding(self):
        df = pd.DataFrame({"text": ["a review"]})
        tokenizer = _make_mock_tokenizer()
        process_dataframe(df, "text", _make_mock_model(["positive"]), tokenizer, "cpu")
        assert tokenizer.call_args[1]["padding"] is True

    def test_uses_id2label_mapping(self):
        df = pd.DataFrame({"text": ["review"]})

        model = MagicMock()
        model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[0.0, 1.0]])
        model.return_value = mock_output

        result = process_dataframe(df, "text", model, _make_mock_tokenizer(), "cpu")
        assert result["Sentiment"].iloc[0] == "positive"

    def test_adds_confidence_column(self):
        df = pd.DataFrame({"text": ["good product", "bad product"]})
        result = process_dataframe(
            df,
            "text",
            _make_mock_model(["positive", "negative"]),
            _make_mock_tokenizer(),
            "cpu",
        )
        assert "Confidence" in result.columns
        assert len(result["Confidence"]) == 2
        for val in result["Confidence"]:
            assert 0.0 <= val <= 1.0

    def test_handles_all_blank_texts(self):
        df = pd.DataFrame({"text": ["", "  ", "\t"]})
        model = MagicMock()
        result = process_dataframe(df, "text", model, MagicMock(), "cpu")

        assert len(result) == 3
        assert all(s == "" for s in result["Sentiment"])
        assert all(c == 0.0 for c in result["Confidence"])
        model.assert_not_called()
        self.mock_progress.progress.assert_called_once_with(1.0)

    def test_handles_mixed_blank_text(self):
        df = pd.DataFrame({"text": ["good product", "", "  ", "bad product"]})
        result = process_dataframe(
            df,
            "text",
            _make_mock_model(["positive", "negative"]),
            _make_mock_tokenizer(),
            "cpu",
        )
        assert result["Sentiment"].iloc[0] == "positive"
        assert result["Sentiment"].iloc[1] == ""
        assert result["Confidence"].iloc[1] == 0.0
        assert result["Sentiment"].iloc[2] == ""
        assert result["Confidence"].iloc[2] == 0.0
        assert result["Sentiment"].iloc[3] == "negative"

    def test_confidence_values_in_valid_range(self):
        df = pd.DataFrame({"text": [f"text {i}" for i in range(5)]})
        result = process_dataframe(
            df,
            "text",
            _make_mock_model(
                ["positive", "negative", "positive", "negative", "positive"]
            ),
            _make_mock_tokenizer(),
            "cpu",
        )
        for val in result["Confidence"]:
            assert 0.0 <= val <= 1.0
