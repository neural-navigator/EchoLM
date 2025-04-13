import json
from datasets import load_dataset
import re

dataset_name = "Trelis/tiny-shakespeare"
dataset = load_dataset(dataset_name)
text = "\n".join(dataset["train"]["Text"])

def tokenize(txt):
    txt = txt.lower()
    tokens = re.findall(r"\b\w+\b", txt)
    return tokens

tokens = tokenize(text)

word_dict = dict(enumerate(set(tokens)))
token_dict = {k: v for v, k in word_dict.items()}

training_data = [token_dict[i] for i in tokens]

with open("../dataset/token_dict.json", "w", encoding="utf-8") as f:
    json.dump(word_dict, f)

with open("../dataset/tokens.json", "w", encoding="utf-8") as f:
    json.dump(training_data, f)
