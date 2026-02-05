# BoW -> BERT : Sentiment Analysis

This project explores sentiment analysis using traditional BoW models
and modern Transformer-based models(BERT).

## Motivation

This project was created to understand the difference between traditional
feature-based NLP methods and modern representation learning approaches.

### (From) Bag-of-Words Pipeline
- CountVectorizer (unigram+bigram)
- Chi-square feature selection
- Logistic Regression

### (To) BERT-based Model
- Pretrained BERT from Huggingface
- FT on sentiment dataset
- Accelerate
- Speed-efficient with JAX/JIT

## Dataset from
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

+++ add Score...

###Citing 🤗BERT
[BERT](https://huggingface.co/google-bert/bert-base-uncased)