import json
import math
import torch
from pathlib import Path
from tqdm import tqdm
from model_code.manager import ChatManager
from quant_tool.save_load import load_quant_model

all_data = [
    json.loads(line)['inputs_pretokenized']
    for file in Path("CEval/val").rglob("*.jsonl")
    for line in file.read_text().splitlines()
    if len(line)
]

batch_size = 4
all_data = [
    all_data[idx: idx + batch_size]
    for idx in range(0, len(all_data), batch_size)
]

model, tokenizer = load_quant_model('../model_file/temp.pth', '../model_file/sentencepiece.model', torch_dtype=torch.float16)

manager = ChatManager(config=None, model=model, tokenizer=tokenizer, device='cuda')

losses = []
progress_bar = tqdm(all_data)

for texts in tqdm(all_data):

    input_ids, labels = manager.tokenizer(texts, padding=True, max_length=2048, return_labels=True)

    with torch.no_grad():

        loss, _, _ = manager.model(input_ids.to('cuda'), all_kv_cache=None, labels=labels.to('cuda'))
        losses.append(loss)

avg = sum(losses)/len(losses)
print(f'ppl:{math.exp(avg):.6f}')
