import os
import re
import json
from collections import defaultdict
import math

import numpy as np


class NGram():
    def __init__(self, n_params):
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
        n_tokens = len(test_data)
        log_liklihood = 0
        for i in range(n_tokens):
            context = tuple(test_data[:i])
            token = test_data[i]
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
            counts = self.log_probs[n-1].get(context_n)
            if counts:
                return max(counts.items(), key=lambda x: x[1])[0]
        unigram_log = self.log_probs[0].get(())
        if unigram_log:
            return max(unigram_log.items(), key=lambda x: x[1])[0]
        return None

    def predict_next_token_with_temp(self, context, temperature):
        """ This method predicts the next token in the context. We wil prefer the
            longest context present, even if the probability is less than the smaller
            n-gram models """
        if type(context) == str:
            context = re.findall(r"\b[a-zA-Z0-9]+\b|[.]", context.lower())

        if len(context) > self.n_params:
            context = context[-self.n_params:]

        for n in range(len(context), 1, -1):
            context_n = tuple(context[-(n-1):])
            prob = self.log_probs[n-1].get(context_n)
            if prob:
                prob_array = np.array(list(prob.values()))
                logits_t = prob_array / temperature
                exp_logits = np.exp(logits_t - np.max(logits_t))
                prob_val = exp_logits / np.sum(exp_logits)
                return str(np.random.choice(list(prob.keys()), p=prob_val))
        unigram_log = self.log_probs[0].get(())
        if unigram_log:
            return str(max(unigram_log.items(), key=lambda x: x[1])[0])
        return None

    def train(self, data_path):
        training_data = self.handle_data(data_path)
        self._train_ngram(training_data)
        self.get_log_probs()

    def forward(self, context, max_token, temp=0):
        if type(context) == str:
            context = re.findall(r"\b[a-zA-Z0-9]+\b|[.]", context.lower())

        if len(context) > self.n_params:
            context = context[-self.n_params:]

        perplexity = self.calculate_perplexity(context)
        perplexity_dict = {tuple(context): perplexity}
        for _ in range(max_token):
            if temp == 0:
                token = self.predict_next_token(context)
                context.append(token)
            else:
                token = self.predict_next_token_with_temp(context, temp)
                context.append(token)
                changed_perplexity = self.calculate_perplexity(context[-self.n_params:])
                perplexity_dict[token] = changed_perplexity

        return " ".join(context), perplexity_dict
