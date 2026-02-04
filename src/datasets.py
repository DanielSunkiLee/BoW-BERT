import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


#Minimal Pytorch Dataset that BERT can work with
#a
#

class SentimentDataset(Dataset):
    def __init__(self, encodings, labels) :
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx) :
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype = torch.long)
        return item
      
    def __len__(self) :
        return len(self.labels)
    
def split_data(texts, labels, test_size=0.2):
    return train_test_split(
        texts, 
        labels, 
        test_size=test_size, 
        random_state=42
    )