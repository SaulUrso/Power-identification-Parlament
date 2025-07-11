# NLP-project
Project for the HLT course year23/24 @Unipi 
## Table of Contents

- [Dataset Download](#dataset-download)
- [Project Structure](#project-structure)
- [Virtual Environment Setup](#virtual-environment-setup)
- [Experiments](#experiments)
  - [Random Forest](#random-forest)
  - [TF-IDF Embeddings](#tf-idf-embeddings)
  - [Word2Vec Embeddings](#word2vec-embeddings)
  - [FastText Embeddings](#fasttext-embeddings)
  - [Pretrained Word2Vec](#pretrained-word2vec)
  - [RNN Experiments](#rnn-experiments)
- [Executing the Notebooks](#executing-the-notebooks)
- [Fine-Tuned Models](#fine-tuned-models)
- [Requirements](#requirements)

---

## Dataset Download

Download the dataset [here](https://drive.google.com/file/d/1mLmE6wP83fMDvHHUIXYHxaPGhAI1iPNx/view?usp=sharing) and place it in the `Dataset` directory for the notebooks to work.

## Project Structure

```
NLP-project/
├── Dataset/
├── Embeddings/
├── Result/
├── datautils.py
├── preprocessing_utils.py
├── RNNutils.py
├── fine_tuning_utils.py
├── requirements.txt
├── TF-IDF embeddings.ipynb
├── w2w_embeddings.ipynb
├── fast_text.ipynb
├── pre trained w2v.ipynb
├── RNN_experiments.ipynb
└── finetuning-politics.ipynb
```

## Virtual Environment Setup

1. Create a virtual environment:
   ```bash
   python -m venv myenv
   ```
2. Activate the environment:
   - **Windows:**  
     `myenv\Scripts\activate`
   - **macOS/Linux:**  
     `source myenv/bin/activate`
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
4. Verify installation:
   ```bash
   pip list
   ```
5. Deactivate when done:
   ```bash
   deactivate
   ```

**Notes:**
- The environment does **not** include PyTorch packages for CUDA.
- If you have dependency issues, try Python 3.11:
  ```bash
  py -3.11 -m venv myenv
  # or
  python3.11 -m venv myenv
  ```
- For spaCy pipelines, install them with:
  ```bash
  python -m spacy download en_core_web_sm
  python -m spacy download en_core_web_md
  python -m spacy download en_core_web_lg
  ```
- Upgrading pip before installing requirements may help:
  ```bash
  python -m pip install --upgrade pip
  ```

## Experiments

### Random Forest

Run `Random_Forest_Final.ipynb` with `preprocessing_utils.py` in the same folder.

### TF-IDF Embeddings

- Create TF-IDF embeddings and evaluate with logistic regression.
- Results of grid search and test set evaluation are saved to files.

### Word2Vec Embeddings

- Create Word2Vec embeddings and evaluate with logistic regression.
- Grid search over context window and vector size.
- Saves models and results for each configuration.
- Computes word analogy for the best model.

### FastText Embeddings

- Similar to Word2Vec, but with FastText embeddings.
- Grid search over context window and vector size.
- Saves models and results for each configuration.
- Computes word analogy for the best model.

### Pretrained Word2Vec

- Test different pretrained models from Gensim with logistic regression.
- Grid search over logistic regression hyperparameters.
- Fine-tuning for Word2Vec and FastText using different tokenizers (`punkt`, `bert_based_uncased`).
- Results are provided for validation set only.

### RNN Experiments

- Requires `RNNutils.py` and `datautils.py` in the same folder.
- Loads and splits dataset, saves splits.
- Optionally downloads pretrained embeddings.
- Trains RNN models, logs metrics, and saves results to CSV.
- Visualizes and evaluates model performance.

## Executing the Notebooks

For `TF-IDF embeddings.ipynb`, `w2w_embeddings.ipynb`, `fast_text.ipynb`, and `pre trained w2v.ipynb`, ensure the following:
- Folders: `Result/` and `Embeddings/`
- File: `datautils.py` in the same directory

## Fine-Tuned Models

- Fine-tuning is demonstrated in a [Kaggle notebook](https://www.kaggle.com/code/saulurso/finetuning-politics).
- A copy of `finetuning-politics.ipynb` is included for reference (may not run locally).
- Requires `fine_tuning_utils.py`.
- The Kaggle environment differs from `requirements.txt`.

## Requirements

All dependencies are listed in `requirements.txt`.

---