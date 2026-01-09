
# Spam Detector

This repository contains a Jupyter notebook (`spam_detector.ipynb`) that trains a binary spam classifier using a BERT-based model and the `mshenoda/spam-messages` dataset from Hugging Face.

**Contents**
- `spam_detector.ipynb`: Notebook with data loading, preprocessing, model setup, training and evaluation.

**Quick overview**
- Dataset: `mshenoda/spam-messages` (loaded via `datasets.load_dataset`).
- Model: `google-bert/bert-base-uncased` used with `AutoModelForSequenceClassification` (2 labels).
- Preprocessing: undersampling to balance classes, tokenization with BERT tokenizer, truncation/padding to `max_length=512`.
- Training: `transformers.Trainer` with `TrainingArguments` (3 epochs, learning rate 2e-5, `adamw_torch`).
- Metrics: accuracy, F1, ROC AUC, recall.

## Requirements
Run the notebook in a Python environment with the following packages installed (examples below):

```bash
pip install --upgrade pip
pip install numpy datasets transformers torch torchvision accelerate scikit-learn
```

Notes:
- A recent Python 3.8+ is recommended.
- GPU is recommended for faster training (install appropriate CUDA-enabled `torch` if needed).

## Dataset access
- The notebook loads `mshenoda/spam-messages` from the Hugging Face Hub using `datasets.load_dataset`.
- If the dataset requires authentication, login with `huggingface-cli login` before running the notebook.

## Running the notebook
1. Open `spam_detector.ipynb` in Jupyter, JupyterLab, or Colab.
2. Ensure dependencies are installed (see Requirements).
3. Run cells top-to-bottom. The notebook does:
	- Install required packages (some cells use `%pip install`).
	- Load and undersample the dataset for class balance.
	- Map labels to integer ids and tokenize texts.
	- Initialize `AutoModelForSequenceClassification` and `Trainer`.
	- Train the model with `trainer.train()` and evaluate on validation/test sets.

## Key implementation notes
- Undersampling: the notebook balances classes by undersampling to the smallest class size (`del_undersample`).
- Label mapping: labels are mapped to integers using a deterministic `label2id` derived from the dataset labels.
- Tokenization: `truncation=True`, `padding='max_length'`, `max_length=512`.
- Training arguments (editable in the notebook): `learning_rate=2e-5`, `per_device_train_batch_size=16`, `num_train_epochs=3`, `weight_decay=0.01`.

## Evaluation
- The notebook computes `accuracy`, `f1`, `auc`, and `recall` via `sklearn.metrics`.

## Troubleshooting
- If you run into CUDA/torch issues, verify your `torch` installation matches your CUDA version (see https://pytorch.org/).
- If dataset download fails, confirm Hugging Face credentials (`huggingface-cli login`) and network access.

## Next steps
- Convert training to a standalone script for command-line runs.
- Add model checkpointing and early stopping.
- Add a small inference script for single-message predictions.