# SiEBERT Pipeline

Streamlit app for classifying text sentiment as positive/negative using [SiEBERT](https://huggingface.co/siebert/sentiment-roberta-large-english) (RoBERTa-large, trained on ~1.4M diverse texts).

## Features

- Upload a CSV or try built-in sample data
- Auto-detects text columns with manual override
- Binary sentiment (positive/negative) with confidence scores
- Summary metrics: total rows, positive/negative counts, average confidence
- Results table with CSV download
- Batched GPU/MPS/CPU inference in float16
- Handles empty, whitespace-only, and malformed input

## Setup

```bash
uv sync
```

Set a [Hugging Face token](https://huggingface.co/settings/tokens) for authenticated model downloads:

```bash
export HF_TOKEN=hf_...
```

## Usage

```bash
uv run streamlit run streamlit_app.py
```

## Testing

```bash
uv run pytest
```

## Citation

If you use SiEBERT in your work, please cite the following paper:

> Hartmann, J., Heitmann, M., Siebert, C., & Schamp, C. (2023). More than a Feeling: Accuracy and Application of Sentiment Analysis. *International Journal of Research in Marketing*, 40(1), 75-87.

```bibtex
@article{hartmann2023,
  title = {More than a Feeling: Accuracy and Application of Sentiment Analysis},
  journal = {International Journal of Research in Marketing},
  volume = {40},
  number = {1},
  pages = {75-87},
  year = {2023},
  doi = {https://doi.org/10.1016/j.ijresmar.2022.05.005},
  url = {https://www.sciencedirect.com/science/article/pii/S0167811622000477},
  author = {Jochen Hartmann and Mark Heitmann and Christian Siebert and Christina Schamp},
}
```
