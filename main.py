import torch
from model_code.manager import ChatManager
from model_code.tokenizer import chat_template

manager = ChatManager.from_pretrained("model_file", device=torch.device("cpu"))

prompt = chat_template([], "你是谁？")
print(prompt)

for text in manager.generate(prompt):

    print(text)
