<h1 align="center">BoW2BERT : Sentiment Analysis</h1>

This project explores sentiment analysis using traditional Bag-of-Words(BoW) models
and modern Transformer-based models,BERT. This project was motivated from adaption of one of the nowadays' serious standard model, Transformer, especially BERT to feature-based NLP methods, Bag-of-Words.

### (From) Bag-of-Words Pipeline
- CountVectorizer (unigram+bigram)
- Chi-square feature selection
- Logistic Regression

### (To) BERT-based Model
- Pretrained BERT from Huggingface
- FT on sentiment dataset
- Accelerate
- Speed-efficient with JAX/JIT

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

+++ add Score...

### Citing 🤗BERT
[BERT](https://huggingface.co/google-bert/bert-base-uncased)