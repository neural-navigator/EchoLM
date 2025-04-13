import json
import math
from collections import defaultdict

import torch
import torch.nn as nn


class BasicLLM(nn.Module):
    def __init__(self, n_params=10):
        self.n_params = n_params
        self.state = [{} for _ in range(n_params)]
        # self.num_train_tokens = len(self.train_data)
        self.training_dataset_path = "../dataset/tokens.json"
        self.log_probs = []

    def get_data(self):
        with open(self.training_dataset_path, "r", encoding="utf-8") as f:
            self.training_data = json.load(f)

    def train(self):
        for i in range(1, self.n_params+1):
            for item in range(len(self.training_data)):
                if i + item >= len(self.training_data):
                    break
                context = tuple(self.training_data[item: item+i])
                next_token = self.training_data[item+i]
                if context not in self.state[i-1]:
                    self.state[i-1][context] = defaultdict(int)
                self.state[i-1][context][next_token] = self.state[i-1][context][next_token] + 1

    def compute_laplace_log_probs(self):
        for level in self.state:
            level_logprob = {}
            for context, nex_token in level.items():
                total_count = sum(nex_token.values())
                level_logprob[context] = {}
                for token, count in nex_token.items():
                    prob = (count+1) / (total_count+len(set(self.training_data)))
                    level_logprob[context][token] = math.log(prob)
            self.log_probs.append(level_logprob)

    def run(self):
        self.get_data()
        self.train()
        self.compute_laplace_log_probs()
        with open("../dataset/trained_model.json", "w", encoding="utf-8") as f:
            json.dump(self.state, f)


if __name__ == "__main__":
    basic = BasicLLM()
    basic.run()
