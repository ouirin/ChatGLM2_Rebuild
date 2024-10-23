import time
import torch
from model_code.manager import ChatManager
from quant_tool.save_load import save_quant_model
from quant_tool.quant_util import bind_quantizer, replace_linear


def quant_model(model, linear_bit=4, linear_group=32):

    qlayers = bind_quantizer(model, linear_bit=linear_bit, linear_group=linear_group)

    replace_linear(model, qlayers)

    return model


manager = ChatManager.from_pretrained("../model_file", device=torch.device("cpu"))
print(manager.model)

start = time.time()
model_q = quant_model(model=manager.model)
end = time.time()
print(f"time cost：{end-start}")
print(model_q)

save_quant_model(model_q, "../model_file/temp.pth")
