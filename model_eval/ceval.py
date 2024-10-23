import time
import json
import glob
import torch
from tqdm import tqdm
from model_code.manager import ChatManager
from quant_tool.save_load import load_quant_model

model, tokenizer = load_quant_model('../model_file/temp.pth', '../model_file/sentencepiece.model', torch_dtype=torch.float16)

manager = ChatManager(config=None, model=model, tokenizer=tokenizer, device='cuda')

choices = ["A", "B", "C", "D"]
choice_tokens = [tokenizer.encode(choice, add_special_tokens=False)[0] for choice in choices]
extraction_prompt = '综上所述，ABCD中正确的选项是：'


def build_prompt(text):
    return "[Round {}]\n\n问：{}\n\n答：".format(1, text)


accuracy_dict, count_dict = {}, {}
output_length = []
input_output_length = []
cost_time = []

with torch.no_grad():

    for entry in glob.glob("/CEval/val/**/*.jsonl", recursive=True):

        dataset = []

        with open(entry, encoding='utf-8') as file:

            for line in file:

                dataset.append(json.loads(line))

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

        correct = 0

        for batch in tqdm(dataloader):

            texts = batch["inputs_pretokenized"]
            queries = [build_prompt(query) for query in texts]
            input_ids = tokenizer(queries, padding=True, max_length=2048).to('cuda')

            start_time = time.time()
            outputs = manager.batch_generate(input_ids, max_generated_tokens=512)
            torch.cuda.empty_cache()
            end_time = time.time()

            cost_time.append(end_time - start_time)
            input_output_length.append((input_ids.shape[1]+outputs.shape[1]) * outputs.shape[0])
            output_length.append(outputs.shape[1] * outputs.shape[0])
            speed1 = round(sum(cost_time) / sum(input_output_length) * 1000, 2)
            speed2 = round(sum(cost_time) / sum(output_length) * 1000, 2)
            print(f'speed output: {speed2}, speed total: {speed1}')

            intermediate_outputs = []
            for idx in range(len(outputs)):
                output = outputs.tolist()[idx]
                response = tokenizer.decode(output)
                intermediate_outputs.append(response)

            texts = [text + intermediate + "\n" + extraction_prompt for text, intermediate in zip(texts, intermediate_outputs)]
            queries = [build_prompt(query) for query in texts]
            input_ids = tokenizer(queries, padding=True, max_length=2048).to('cuda')

            outputs = manager.batch_generate(input_ids, return_last_logit=True)
            logits = outputs[:, -1]
            logits = logits[:, choice_tokens]
            preds = logits.argmax(dim=-1)
            correct += (preds.cpu() == batch["label"].cpu()).sum().item()

        accuracy = correct / len(dataset)
        data_amount = len(dataset)
        print(entry, data_amount, accuracy)

        accuracy_dict[entry] = accuracy
        count_dict[entry] = len(dataset)

acc_total, count_total = 0.0, 0
stem_total, stem_count_total = 0.0, 0
social_sciences_total, social_sciences_count_total = 0.0, 0
humanities_total, humanities_count_total = 0.0, 0
others_total, others_count_total = 0.0, 0

for key in accuracy_dict:
    acc_total += accuracy_dict[key] * count_dict[key]
    count_total += count_dict[key]

for key in accuracy_dict:

    if "STEM" in key:
        stem_total += accuracy_dict[key] * count_dict[key]
        stem_count_total += count_dict[key]

    if "Social_Science" in key:
        social_sciences_total += accuracy_dict[key] * count_dict[key]
        social_sciences_count_total += count_dict[key]

    if "Humanities" in key:
        humanities_total += accuracy_dict[key] * count_dict[key]
        humanities_count_total += count_dict[key]

    if "Other" in key:
        others_total += accuracy_dict[key] * count_dict[key]
        others_count_total += count_dict[key]

print(stem_total / stem_count_total)
print(social_sciences_total / social_sciences_count_total)
print(humanities_total / humanities_count_total)
print(others_total / others_count_total)
print(acc_total / count_total)
