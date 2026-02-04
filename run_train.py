from pathlib import Path
import torch
from src import load_dataset
from src import split_data, SentimentDataset
from src import load_model, load_tokenizer
from src import compute_metrics
from src import train_model
from accelerate import Accelerator
accelerator = Accelerator()

#set up csv file path.
#
#
BASE_DIR = Path(__file__).resolve().parent
ML_DIR = BASE_DIR.parent.parent

DATA_DIR = (
    ML_DIR
    / "Representation_Learning"
    / "Representation-Learning-for-NLP"
    / "data"
)

CSV_PATH = DATA_DIR / "training.1600000.processed.noemoticon.csv"

#CSV file
#  ↓
#pandas DataFrame
#  ↓
#df["text"]  ──▶ tokenizer ──▶ input_ids + attention_mask
#df["label"] ─────────────────▶ labels


# load data
texts, labels = load_dataset(CSV_PATH)

# split
text_train, text_val, y_train, y_val = split_data(texts, labels)

# model & tokenizer
tokenizer = load_tokenizer()
model = load_model()
optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-5)
# tokenize
train_enc = tokenizer(text_train, truncation=True, padding=True, max_length=128)
val_enc = tokenizer(text_val, truncation=True, padding=True, max_length=128)

train_dataset = SentimentDataset(train_enc, y_train)
val_dataset = SentimentDataset(val_enc, y_val)

# train
trainer = train_model(model, train_dataset, val_dataset, compute_metrics)
loss_function = torch.nn.CrossEntropyLoss()

optimizer, train_dataset, val_dataset, trainer = accelerator.prepare(
    optimizer, train_dataset, val_dataset, trainer)

for batch in train_dataset :
    optimizer.zero_grad()
    inputs, targets = batch
    outputs = trainer.model(**inputs)
    loss = loss_function(outputs, targets)
    accelerator.backward(loss)
    optimizer.step()

tokenizer.save_pretrained("./saved_model")