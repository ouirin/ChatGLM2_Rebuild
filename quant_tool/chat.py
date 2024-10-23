import torch
from model_code.manager import ChatManager
from model_code.tokenizer import chat_template
from quant_tool.save_load import load_quant_model

model, tokenizer = load_quant_model('../model_file/temp.pth', '../model_file/sentencepiece.model', torch_dtype=torch.float16)

manager = ChatManager(config=None, model=model, tokenizer=tokenizer, device='cuda')

prompt = chat_template([], "你是谁？")
print(prompt)

for text in manager.generate(prompt):

    print(text)

