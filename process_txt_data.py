from transformers import GPT2Tokenizer
import os
import glob
import tqdm
import numpy as np
from torch.utils.data import DataLoader, Dataset, IterableDataset
import torch

'''
fin = open("amfamdata1.txt", 'r')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizedData = []
read_data = fin.readlines()

for line in read_data:
    encodedData.append(tokenizer(line))
'''
#print(len(read_data))


class AmfamDataset(Dataset): #basically LineByLineTextDataset
    def __init__(self, tokenizer, file_path, block_size):
        #super(AmfamDataset).__init__()
        with open(file_path, encoding = "utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0) and not line.isspace()]

        batch_encoding = tokenizer(lines, add_special_tokens = True, truncation = True, max_length = block_size)
        self.dataset = batch_encoding['input_ids']
        self.dataset = [{"input_ids":torch.tensor(e, dtype = torch.long)} for e in self.dataset] #don't understand why we need this as a list of dicts

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i]




'''

afdata = torch.load('amfamdata1.txt')




def load_dataset(tokenizer, path, combine):
    paths = []
    if os.path.isfile(path):
        paths.append(path)
    elif os.path.isdir(path):
        for (dirpath, _, fnames) in os.walk(path):
            for fname in fnames:
                paths.append(os.path.join(dirpath, fname))
    else:
        paths = glob.glob(path)

    token_chunks = []
    raw_text = ''

    for path in tqdm.tqdm(paths):
        with open(path, 'r', encoding = 'utf8', errors = 'ignore') as fp:
            raw_text = fp.readlines()[0:1000]
        if len(raw_text) >= combine:
            print("yes")
            tokens = np.stack(tokenizer(raw_text, return_tensors="pt", padding = True))
            print(tokens)
            token_chunks.append(tokens)
            raw_text = ''
        else:
            print("no")
            raw_text += '<|endoftext|>'

    if raw_text:
        tokens = np.stack(tokenizer(raw_text))
        token_chunks.append(tokens)
    return token_chunks

tokenizer.add_special_tokens({'pad_token': '[PAD]'})
'''
#print(load_dataset(tokenizer, 'amfamdata1.txt', 1000))
