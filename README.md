# danish-edu-llm-classifier

A comprehensive toolkit designed to **enhance multilingual AI capabilities** by **analyzing and modeling educational content**, particularly within a **low-resource language context (Danish)** using the **FineWeb datasets**. This project focuses on understanding **human annotation quality and inter-annotator agreement** and developing **scalable, consistent content classification pipelines** through the **fine-tuning of encoder models** and investigation of **cross-lingual transfer strategies**.
For the paper on this git, go to `docs/FP_25`.

---

## Project Structure

```
danish-edu-llm-classifier/
├── data/                  # Raw, interim, and processed datasets
├── docs/
│   └── visualizations/    # Generated plots and figures for reports
├── notebooks/             # Jupyter notebooks for analysis and exploration
├── src/
│   ├── annotation/        # Annotation tools and scripts
│   ├── data_processing/   # Data processing and merging scripts
│   ├── evaluation/        # Evaluation utilities and metrics
│   └── training/          # Model training scripts and configs
├── evaluation/
│   ├── results/           # Inference and evaluation outputs (not tracked by git)
│   └── plots/             # Generated evaluation plots (not tracked by git)
├── archive/               # Deprecated/legacy scripts and results
├── models/                # Saved models (if any)
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## Data
- All datasets are in `data/` (with subfolders for `raw/`, `interim/`, and `processed/`).
- Use scripts in `src/data_processing/` to prepare and merge datasets.

## Training
- Main training scripts are in `src/training/`.
- Configurations are in `src/training/config/`.
- Example: `python src/training/train.py src/training/config/base.yaml`

## Annotation
- Manual annotation tool: `src/annotation/annotation.py` (Streamlit app)
- Annotation guidelines and example files are in `src/annotation/`
- Run with: `streamlit run src/annotation/annotation.py`

## Evaluation & Inference
- Run inference and evaluation scripts from `evaluation/`.
- Results and plots are saved in `evaluation/results/` and `evaluation/plots/` (these are git-ignored).
- Notebooks for analysis are in `notebooks/`.

## Visualization & Reporting
- All generated figures and summary tables for reports are in `docs/visualizations/`.

## Archive
- Old experiments, scripts, and results are kept in `archive/` for reference.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Prepare data as needed using scripts in `src/data_processing/`.
3. Train models using scripts in `src/training/`.
4. Annotate or evaluate as needed.

---

## Notes
- For legacy scripts and results, see the `archive/` directory.

---
