# vLLM Prompting for Cardiovascular Mortality Research

This repository contains a **public adaptation of my original vLLM prompting notebook** from my 2026 University of Luxembourg Master's thesis, *A Comparative Study of Fine-Tuning Techniques for Large Language Models with a Focus on Cardiovascular Risk Prediction*.

The notebook demonstrates the original code structure for **zero-shot and few-shot prompting, Hugging Face tokenizer setup, vLLM batch inference, percentage-response parsing, and ROC-AUC evaluation**. The private ATTICA data-loading and prompt-generation dependencies have been replaced with independently written **synthetic input examples**. The model inference code has not been replaced with mock predictions.

## Notebook

[`ATTICA_vLLM_public.ipynb`](ATTICA_vLLM_public.ipynb)

The notebook requires Python and, to execute real inference, a compatible **vLLM**, **Transformers**, **NumPy**, **Pandas**, **scikit-learn**, **tqdm** and a GPU able to host the chosen model. Weights & Biases is optional. Open it in Jupyter and execute top to bottom. The original MedGemma model is large; users may select a smaller compatible instruction-tuned model. This adapted notebook has been syntax-checked but **not GPU-tested**.

## Research context

The original thesis compared LLM prompting and Med42-8B fine-tuning against classical machine learning baselines for **long-term cardiovascular mortality prediction**.

Fine-tuned Med42-8B achieved a ROC-AUC of **0.9225** on the main near-balanced evaluation subset, with performance comparable to classical ML approaches. The research also investigated:

- **Evaluation reliability:** Identified an epoch-selection bias that inflated ROC-AUC by approximately 0.012–0.024.
- **Generative scoring:** Improved ROC-AUC from 0.803 to 0.921 by using token probabilities instead of decoded predictions.
- **Training efficiency:** Achieved approximately 3× faster training using two GPUs.

These findings come from the original thesis and are **not reproduced by the public synthetic-data notebook**.

## Experimental results

The original thesis evaluated Med42-8B using stratified five-fold cross-validation across three random seeds. The TRIPLE_BEST protocol used an internal validation split for epoch selection, keeping the outer test folds separate from model selection.

| Random seed | Mean ROC-AUC |
|---|---|
| 42 | 0.9179 |
| 43 | 0.9290 |
| 44 | 0.9216 |
| **Mean** | **0.9228** |

These results demonstrate the consistency of the fine-tuned model across repeated evaluations. They were recorded during the original thesis experiments and are not reproduced by the public prompting notebook.

## Publication scope

The original ATTICA participant data and private University of Luxembourg implementation cannot be shared. This notebook contains independently constructed synthetic records and prompts instead; it includes no participant-level outputs.
