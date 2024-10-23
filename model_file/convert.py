import json
import shutil
import torch
from pathlib import Path
from tqdm.auto import tqdm
from collections import OrderedDict
from safetensors.torch import save_file
from model_code.loader import ChatGLMLoadConfig

src_path = Path("./chatglm2")
dst_path = Path("./ChatGLM2_Rebuild/model_file")

name_mapping = {
    'transformer.embedding.word_embeddings.weight': 'word_embedding.weight',
    'transformer.encoder.final_layernorm.weight': 'final_ln.weight',
    'transformer.output_layer.weight': 'lm_head.weight'
}

for i in range(28):
    name_mapping.update({
        f'transformer.encoder.layers.{i}.input_layernorm.weight': f'layers.{i}.attn_ln.weight',
        f'transformer.encoder.layers.{i}.self_attention.query_key_value.weight': f'layers.{i}.attn.qkv_proj.weight',
        f'transformer.encoder.layers.{i}.self_attention.query_key_value.bias': f'layers.{i}.attn.qkv_proj.bias',
        f'transformer.encoder.layers.{i}.self_attention.dense.weight': f'layers.{i}.attn.o_proj.weight',
        f'transformer.encoder.layers.{i}.post_attention_layernorm.weight': f'layers.{i}.ffn_ln.weight',
        f'transformer.encoder.layers.{i}.mlp.dense_h_to_4h.weight': f'layers.{i}.ffn.w_in.weight',
        f'transformer.encoder.layers.{i}.mlp.dense_4h_to_h.weight': f'layers.{i}.ffn.w_out.weight',
    })


indices = json.loads((src_path / "pytorch_model.bin.index.json").read_bytes())
bin_files = set(indices["weight_map"].values())

for bin_file in tqdm(bin_files):

    state_dict = torch.load(src_path / bin_file, map_location=torch.device("cpu"))
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():

        if k not in name_mapping:
            continue
        new_state_dict[name_mapping[k]] = v

    save_file(new_state_dict, dst_path / bin_file.replace(".bin", ".safetensors"))

config = ChatGLMLoadConfig(
    weight_files=[bin_file.replace(".bin", ".safetensors") for bin_file in bin_files],
    torch_dtype="bfloat16",
)

shutil.copy(src_path / "tokenizer.model", dst_path / config.tokenizer_file)

config_path = dst_path / "config.json"
config_path.write_text(config.to_json())

