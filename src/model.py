from collections import defaultdict
import math
import re
import torch
import torch.nn as nn


class NGram(nn.Module):
    def __init__(self, n_params):
        super(NGram, self).__init__()
        self.n_params = n_params
        self.state = [{} for _ in range(n_params)]
        self.vocab_size = 0
        self.log_probs = []

    def handle_data(self, data_path):
        """This method read the data file"""
        with open(data_path) as f:
            content = f.read()
        content = re.findall(r"\b[a-zA-Z0-9]+\b|[.]", content.lower())
        self.vocab_size += len(set(content))
        return content

    def _train_ngram(self, training_data):
        """This method trains the model"""
        for i in range(self.n_params):
            for idx, j in enumerate(training_data):
                if idx + i >= len(training_data):
                    break
                context = tuple(training_data[idx: idx + i])
                next_token = training_data[idx + i]
                if context not in self.state[i]:
                    self.state[i][context] = defaultdict(int)
                self.state[i][context][next_token] = self.state[i][context][next_token] + 1

    def get_log_probs(self):
        """This method returns the log probabilities of each token in training data"""
        for item in self.state:
            log_dict = {}
            for k, v in item.items():
                total_count = sum(v.values())
                log_dict[k] = {k1: math.log((v1 + 1) / (total_count + self.vocab_size))
                               for k1, v1 in v.items()}
            self.log_probs.append(log_dict)

    def calculate_perplexity(self, test_data):
        """This method calculates the perplexity of the model"""
        test_tokens = test_data.lower().split()
        n_tokens = len(test_tokens)
        log_liklihood = 0
        if n_tokens > self.n_params:
            context = test_data[-self.n_params:]
        else:
            context = test_data
        for i in range(n_tokens):
            context = tuple(test_tokens[:i])
            token = test_tokens[i]
            log_liklihood += self.log_probs[i].get(context, {}).get(token, math.log(1e-10))
        avg_log_liklihood = log_liklihood / n_tokens
        perplexity = math.exp(-avg_log_liklihood)
        return perplexity

    def predict_next_token(self, context):
        """ This method predicts the next token in the context. We wil prefer the
        longest context present, even if the probability is less than the smaller
        n-gram models """
        if type(context) == str:
            context = re.findall(r"\b[a-zA-Z0-9]+\b|[.]", context.lower())

        if len(context) > self.n_params:
            context = context[-self.n_params:]

        for n in range(len(context), 1, -1):
            context_n = tuple(context[-(n-1):])
            counts = self.state[n-1].get(context_n)
            if counts:
                return max(counts.items(), key=lambda x: x[1])[0]
        unigram_counts = self.state[0].get(())
        if unigram_counts:
            return max(unigram_counts.items(), key=lambda x: x[1])[0]
        return None

    def generate_text(self, context, max_token):
        pass

