from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from torch.utils.data import DataLoader, Dataset, IterableDataset
import torch


class AmfamDataset(Dataset): #basically LineByLineTextDataset
    def __init__(self, tokenizer, file_path, block_size):
        super(AmfamDataset).__init__()
        with open(file_path, encoding = "utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0) and (len(line) < 256) and not line.isspace()]

        batch_encoding = tokenizer(lines, add_special_tokens = True, truncation = True, max_length = block_size)
        self.dataset = batch_encoding['input_ids']
        self.dataset = [{"input_ids":torch.tensor(e, dtype = torch.long)} for e in self.dataset] #don't understand why we need this as a list of dicts

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i]



tokenizer = GPT2Tokenizer.from_pretrained("gpt2", pad_token = '<pad>')

path = "amfamdata1.txt"

dataset = AmfamDataset(tokenizer=tokenizer, file_path = path, block_size = 128)
print(dataset(6))
