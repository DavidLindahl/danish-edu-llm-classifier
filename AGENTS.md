# Repository Guidelines

This repository contains scripts and configuration files for building a multilingual educational content classifier focused on Danish data.  The main folders are:

- `data_processing/` – scripts for downloading, filtering and labelling datasets.
- `training/` – training utilities and configuration YAML files.
- `self_annotation/` – a small Streamlit tool used for manual annotation.
- `archive/` – deprecated or experimental code kept for reference.

## Coding style

- Use **Python 3.10+**.
- Follow **PEP8** with 4‑space indentation.
- Provide docstrings for modules, classes and functions when possible.
- Keep script entry points guarded with `if __name__ == "__main__":`.
- Configuration files are written in YAML.  Preserve formatting when editing them.

## Contribution tips

- Describe any new datasets or scripts in `README.md`.
- Update `requirements.txt` when adding new dependencies.
- Avoid committing large raw data files.
- Keep commit messages short and descriptive.

