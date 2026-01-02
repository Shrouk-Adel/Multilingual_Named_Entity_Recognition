# Multilingual Named Entity Recognition (NER) with XLM-RoBERTa

This project implements and evaluates a Named Entity Recognition (NER) model using the **XLM-RoBERTa (Cross-lingual Language Model)** architecture. The goal is to demonstrate the effectiveness of multilingual models for cross-lingual transfer learning, where a model trained on one or two high-resource languages can perform well on new, unseen languages.

🚀 Live Demo
You can try out the trained model directly in your browser using the public demo on Hugging Face Spaces:

**[Try the XLM-RoBERTa NER Demo](https://huggingface.co/spaces/shroukAdel/xlm-roberta-ner-demo)**

## 🌟 Overview

The project involves the following key steps:

1. **Dataset Loading and Preprocessing:** Utilizing the **PAN-X** dataset for multiple languages (German, French, Italian, and English).
2. **Multilingual Tokenization:** Implementing a subword tokenization used by XLM-RoBERTa.
3. **Model Training:** Fine-tuning the `xlm-roberta-base` model for the token classification task using the Hugging Face `Trainer` API.
4. **Cross-Lingual Evaluation:** Running two main experiments:
* **Zero-shot transfer:** Training on a single language (German) and evaluating on French, Italian, and English.
* **Joint Training:** Training on multiple languages (e.g., German + French) to improve performance across the board.



## 🛠️ Installation

This project is implemented as a Jupyter notebook and requires Python. You can set up the environment using `pip`:

```bash
# Clone the repository (if applicable)
# git clone <your-repo-link>
# cd Multilingual_NER_Project

# Install necessary libraries
!pip install datasets transformers seqeval accelerate -U

```

The notebook was run on a Google Colab environment using a **T4 GPU**.

## 📊 Dataset

The model is trained and evaluated on a subset of the **PAN-X** (Cross-lingual NER) dataset, which provides NER annotations across many languages.


### Named Entity Tags

The model is trained to identify the standard NER tags:

* **`O`**: Outside of a named entity.
* **`B-PER` / `I-PER**`: Beginning/Inside a Person entity.
* **`B-ORG` / `I-ORG**`: Beginning/Inside an Organization entity.
* **`B-LOC` / `I-LOC**`: Beginning/Inside a Location entity.
* **`B-MISC` / `I-MISC**`: Beginning/Inside a Miscellaneous entity.

## 🧠 Model and Training

### Architecture

* **Pretrained Model:** `xlm-roberta-base`
* **Tokenizer:** `xml-roberta` tokenizer

### Training Configuration

The training utilizes the Hugging Face `Trainer` with the following key arguments:

* **Model Head:** `XLMRobertaForTokenClassification`
* **Learning Rate:** `2e-5`
* **Number of Epochs:** `3`
* **Evaluation Metric:** F1-score (for NER)

## 📈 Experiments and Results

The project explores two main strategies for multilingual NER:

### 1. Zero-Shot Transfer (Trained on German `de`)

The model was trained exclusively on the German (de) dataset and then evaluated on other languages without any further fine-tuning.

| Evaluation Dataset | F1-score |
| --- | --- |
| **French (`fr`)** | 0.703 |
| **Italian (`it`)** | 0.676 |
| **English (`en`)** | 0.602 |

### 2. Joint Training (Trained on German `de` + French `fr`)

The model was trained on a combination of the German and French datasets.

| Evaluation Dataset | F1-score |
| --- | --- |
| **German (`de`)** | 0.866  |
| **French (`fr`)** | 0.860  |
| **Italian (`it`)** | 0.816 (inferred from surrounding cell/output) |
| **English (`en`)** | 0.677|

The joint training approach shows a significant improvement in performance on cross-lingual tasks compared to the zero-shot transfer approach.

### 3. FineTuning on all Language together
| Evaluation Dataset | F1-score |
| --- | --- |
| **German (`de`)** | 0.8682  |
| **French (`fr`)** | 0.8647 |
| **Italian (`it`)** | 0.8575 |
| **English (`en`)** | 0.7870|

Multilingual learning can provide significant gains in performance, especially if the low-resource
languages for cross-lingual transfer belong to similar language families. In our experiments we can see
that erman, French, and Italian achieve similar performance in the all category, suggesting that these
languages are more similar to each other than to English.

## ▶️ Usage

### Model Repository

The final fine-tuned model is publicly available on the Hugging Face Model Hub:

**[xlm-roberta-base-finetuned-panx-all](https://huggingface.co/shroukAdel/xlm-roberta-base-finetuned-panx-all)**

You can easily load the model for inference using the `transformers` library.
