<h1 align="center">BoW2BERT : Sentiment Analysis</h1>

This project explores sentiment analysis using traditional Bag-of-Words(BoW) models
and modern Transformer-based models,BERT. 

The motivation behind this work is to examine the adaption of one of today's most influential standard models -- the Transformer, particulary BERT -- in comparison with classical feature-based NLP approaches such as Bag-of-Words.

By contrasting these methodologies, this project aims to provide a clear perspective on the evolution of sentiment analysis techniques, from conventional statistical representations to deep contextual language models.

### Baseline: Bag-of-Words Pipeline
#### Feature Extraction
- CountVectorizer (unigram+bigram)
- High-dimensional sparse token-frequency matrix
#### Feature Selection
- Chi-square statistical filtering
- Dimensionality reduction via discriminative feature ranking
#### Classifier
- Logistic Regression

### Advanced Model:  BERT-based Architecture
#### Backbone
- Pretrained BERT from Huggingface
- FT on sentiment dataset
#### Optimization & Efficiency
- Hugging Face Accelerate (multi-device & mixed precision training)
- JAX with JIT compilation for improved execution throughput

## Reference 
https://github.com/anujgupta82/Representation-Learning-for-NLP

## How to Run

### 1. Clone Repo
```
git clone https://github.com/DanielSunkiLee/BoW-BERT.git
cd bow-bert
```

### 2. Quickstart
```
pip install -r requirements.txt
```

### 3. Run
```
python train.py
```

## Results

### Training Dynamics (BERT FT)
During fine-tuning, the model demonstrates stable optimization and rapid convergence:

| epoch | Loss | Grad Norm | Learning Rate |
|-------|------|-----------|---------------|
| 3.125 | 0.467|   8.09    |   1.802e-05   |
| 6.25  | 0.092|  0.3828   |   1.602e-05   |
| 9.375 | 0.006| 0.03089   |   1.402e-05   |

### Observations
- Monotonic loss reduction indicating effective FT.
- Gradient norm decay suggests stabilization of parameter updates.
- No signs of gradient explosion or instability.

## Citing 🤗BERT
This project is built upon the pretrained [BERT](https://huggingface.co/google-bert/bert-base-uncased) model from Hugging Face Transformes,leveraging its robust contextual representations for sentiment classification.