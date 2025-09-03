# Learning How to Use Hugging Face [🚧 Project Under Construction 🚧]

Small, self-contained experiments while I learn and test Hugging Face: Transformers, Datasets, and Hub.

## Installation

Recreate the environment and install the main libraries

```bash
conda env create -f environment.yml
```

## Scripts

### Mask Filling: Test a model that fills in masked parts of a sentence

Usage:

* Run with default settings:
```bash
python mask_filling.py
```
* Run with custom text and settings:
```bash
python mask_filling.py --text "Hello! My name is [MASK] and I'm from [MASK], so I speak [MASK]." --temperature 1.1 --n_sentences 5
```

Where:

*  **'text'**: Input text with one or more [MASK] tokens.
    * Example: "I live in [MASK]."
    * Default: "Hello! My name is [MASK]."

* **'temperature'**: Controls randomness in predictions.
    * temperature = 0 → deterministic prediction (always chooses the most likely token).
    * temperature > 0 → adds creativity/diversity (higher = more random).
    * Default: temperature = 0.0

* **'n_sentences'**: Number of completed sentences to generate.
    * Default: n_sentences = 1
