import json
import torch

from pathlib import Path
from tqdm.auto import tqdm
from dataclasses import dataclass, asdict, field
from safetensors.torch import safe_open
from model_code.tokenizer import ChatGLM2Tokenizer
from model_code.model import ChatGLM2Model, ChatGLM2Config


@dataclass
class ChatGLMLoadConfig():

    model_type: ChatGLM2Model = "ChatGLM2Model"
    model_config: ChatGLM2Config = field(default_factory=ChatGLM2Config)
    quant_type: str = "none"
    weight_files: list = field(default_factory=list)
    tokenizer_file: str = "sentencepiece.model"
    torch_dtype: str = "float32"

    def __post_init__(self):
        self.model_config = ChatGLM2Config(**self.model_config)

    def get_torch_dtype(self):
        return getattr(torch, self.torch_dtype)

    @staticmethod
    def from_json(json_str):
        return ChatGLMLoadConfig(**json.loads(json_str))

    def to_json(self):
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


@torch.no_grad()
def load_model_and_tokenizer(model_path):

    model_path = Path(model_path)
    config_path = model_path / "config.json"
    config = ChatGLMLoadConfig.from_json(config_path.read_bytes())

    model = ChatGLM2Model(config.model_config, config.get_torch_dtype())
    state_dict = dict(**model.state_dict())

    for file in tqdm(config.weight_files):

        with safe_open(model_path / file, framework='pt') as f:

            for k in f.keys():

                state_dict[k].copy_(f.get_tensor(k))
                state_dict.pop(k)

    tokenizer = ChatGLM2Tokenizer(model_path / config.tokenizer_file)

    return config, model, tokenizer


