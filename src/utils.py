# src/utils.py
from tqdm import tqdm
import time

def tqdm_test():
    for i in tqdm(range(100), desc="Processing items"):
        time.sleep(0.05)
