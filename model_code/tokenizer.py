import torch
from sentencepiece import SentencePieceProcessor


def chat_template(history, current):

    prompt = ""
    chat_round = 1

    for question, answer in history:

        prompt += f"[Round {chat_round}]\n\n问：{question}\n\n答：{answer}\n\n"
        chat_round += 1

    prompt += f"[Round {chat_round}]\n\n问：{current}\n\n答："

    return prompt


class ChatGLM2Tokenizer:

    def __init__(self, vocab_file):

        self.vocab_file = vocab_file
        self.special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "<sop>", "<eop>"]
        self.text_tokenizer = SentencePieceProcessor(str(vocab_file))
        self.vocab_size = len(self.text_tokenizer) + len(self.special_tokens)
        self.true_vocab_size = len(self.text_tokenizer)

        self.bos_id: int = self.text_tokenizer.bos_id()
        self.eos_id: int = self.text_tokenizer.eos_id()
        self.pad_id: int = self.text_tokenizer.unk_id()

    def __len__(self):
        return self.vocab_size

    def __getitem__(self, key: str):
        if key in self.special_tokens:
            return len(self.text_tokenizer) + self.special_tokens.index(key)
        return self.text_tokenizer[key]

    def encode(self, text, add_special_tokens=True):

        tokens = self.text_tokenizer.encode(text)

        if add_special_tokens:

            tokens = [self["[gMASK]"], self["<sop>"]] + tokens

        return tokens

    def decode(self, text_ids):

        text_ids = list(filter(lambda x: x < self.true_vocab_size, text_ids))
        text = self.text_tokenizer.decode(text_ids)

        return text

    def __call__(self, text, padding=False, max_length=None, return_labels=False):

        input_ids = []
        for t in text:
            input_ids.append(self.encode(t))

        attention_mask = []
        for input_id in input_ids:
            attention_mask.append([1] * len(input_id))

        for i in range(len(input_ids)):
            input_ids[i] = input_ids[i][:max_length]
            attention_mask[i] = attention_mask[i][:max_length]

        max_seq_length = max(map(lambda x: len(x), input_ids))

        if padding == "right":

            for i in range(len(input_ids)):
                pad_length = max_seq_length - len(input_ids[i])
                input_ids[i] = input_ids[i] + pad_length * [self.pad_id]
                attention_mask[i] = attention_mask[i] + pad_length * [0]

        if padding == "left" or padding is True:

            for i in range(len(input_ids)):
                pad_length = max_seq_length - len(input_ids[i])
                input_ids[i] = pad_length * [self.pad_id] + input_ids[i]
                attention_mask[i] = pad_length * [0] + attention_mask[i]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        if return_labels:

            labels = input_ids.masked_fill(~attention_mask.bool(), -100)

            return input_ids, labels

        return input_ids
