import re
import sys

import torch
from model_code.loader import load_model_and_tokenizer


def top_p_sampling(logits, top_k=100, top_p=0.8, temperature=1.0):

    probs = torch.softmax(logits.float() / temperature, dim=-1)
    probs, indices = torch.sort(probs, dim=-1, descending=True)
    probs = probs[..., :top_k]
    indices = indices[..., :top_k]

    cumsum = torch.cumsum(probs, dim=-1)
    probs[(cumsum - probs) > top_p] = 0.0
    probs = probs / torch.sum(probs, dim=-1, keepdim=True)

    next_token = torch.multinomial(probs, num_samples=1)
    output = torch.gather(indices, dim=-1, index=next_token)

    return output[..., 0]


def process_response(response: str):

    response = response.strip()
    response = response.replace("[[训练时间]]", "2023年")
    punkts = [
        [",", "，"],
        ["!", "！"],
        [":", "："],
        [";", "；"],
        ["\?", "？"],
    ]
    for item in punkts:
        response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
        response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)

    return response


class ChatManager():

    def __init__(self, config, model, tokenizer, device=None):

        self.config = config
        self.model = model
        self.tokenizer = tokenizer

        self.device = device
        self.eos_token_id = tokenizer["</s>"]
        self.max_sequence_length = 8192
        self.model.to(self.device)

    @staticmethod
    def from_pretrained(path=None, device=None):

        config, model, tokenizer = load_model_and_tokenizer(path)

        model.to(device=device)

        return ChatManager(config, model, tokenizer, device=device)

    def generate(self, prefix_text, max_generated_tokens=150, top_k=100, top_p=0.8, temperature=1.0):

        prefix_ids = self.tokenizer.encode(prefix_text)

        input_ids = torch.LongTensor([prefix_ids])

        all_kv_cache = None
        generated_tokens = []

        while len(generated_tokens) < max_generated_tokens:

            with torch.no_grad():

                loss, logits, all_kv_cache = self.model(input_ids=input_ids.to(self.device), all_kv_cache=all_kv_cache)

                next_token = top_p_sampling(logits[0, -1], top_k, top_p, temperature).item()

            generated_tokens += [next_token]

            if next_token == self.eos_token_id:
                break

            response_text = process_response(self.tokenizer.decode(generated_tokens))

            if response_text and response_text[-1] != " ":
                yield response_text

            input_ids = torch.tensor([[next_token]]).long()

        return process_response(self.tokenizer.decode(generated_tokens))

    def batch_generate(self, input_ids, max_generated_tokens=512, top_k=100, top_p=0.8, temperature=1.0, return_last_logit=False):

        all_kv_cache = None
        generated_tokens = [[] for _ in range(input_ids.shape[0])]
        finished = [False] * input_ids.shape[0]

        for _ in range(max_generated_tokens):

            with (torch.no_grad()):

                loss, logits, all_kv_cache = self.model(input_ids=input_ids.to(self.device), all_kv_cache=all_kv_cache)

                if return_last_logit == True:
                    return logits

                next_tokens = []

                for i in range(input_ids.shape[0]):

                    next_token = top_p_sampling(logits[i, -1], top_k, top_p, temperature).item()

                    next_tokens.append(next_token)

                    if finished[i] is False:
                        generated_tokens[i] += [next_token]
                    else:
                        generated_tokens[i] += [0]

                    if next_token == self.eos_token_id:
                        finished[i] = True

                if all(finished):
                    break

                input_ids = torch.tensor(next_tokens).unsqueeze(1).long()

        generated_tokens = torch.tensor(generated_tokens)

        return generated_tokens
