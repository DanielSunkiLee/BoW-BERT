<h1 align="center">BoW2BERT : Sentiment Analysis</h1>

## Summary
**BERT outperforms Bag-of-Words by leveraging contextual embeddings, achieving higher accuracy (+4%) and F1-score with stable convergence.**

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
git clone https://github.com/DanielSunkiLee/BoW2BERT.git
cd bow-bert
```

### 2. Quickstart
```
pip install -r requirements.txt
```

### 3. Run
```
python run_train.py
```

## Results

Model Comparison: Bag-of-Words vs BERT 

Baseline: Bag-of-Words
- Validation Accuracy: 0.79
- Evaluation via sentiment_pipeline.score()

🤗 BERT Fine-Tuning (FT)

⏱ Training Setup
- Total steps: 1,000
- Epochs: 31.25
- Training time: ~67 minutes

Training Dynamics
- Loss:
  - 0.459 → 0.056 (early convergence)
  - → ~4e-05 (final epochs)
- Learning Rate:
  - 4.5e-05 → 0 (linear decay)
👉 Indicates stable and effective convergence

| Checkpoint       | Loss  | Accuracy | F1-score |
| ---------------- | ----- | -------- | -------- |
| Mid-training     | 1.353 | 82%      | 0.8125   |
| Final checkpoint | 1.427 | 83%      | 0.8317   |

Analysis
- Final model shows improved Accuracy and F1-score
- Slight increase in loss suggests:
  - minor confidence miscalibration, not severe overfitting
- Overall, the model demonstrates good generalization.

🤔 Key Takaways
- BERT significantly outperforms Bag-of-Words:
  - +4% accuracy gain
  - Better semantic understanding
- Pretrained transformers capture contextual meaning, unlike BoW




### Observations
- Monotonic loss reduction indicating effective FT.
- Gradient norm decay suggests stabilization of parameter updates.
- No signs of gradient explosion or instability.

## Citing 🤗BERT
This project is built upon the pretrained [BERT](https://huggingface.co/google-bert/bert-base-uncased) model from Hugging Face Transformes, leveraging its robust contextual representations for sentiment classification.
