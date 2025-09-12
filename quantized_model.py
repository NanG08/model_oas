from action_model import CNN_GRU  
from torch.quantization import quantize_dynamic
import torch.nn as nn
import torch

model = CNN_GRU()
model.load_state_dict(torch.load("model_oas\\best_model.pt", map_location="cpu"))
model.eval()

print("[INFO] Original model loaded. Starting quantization...")

quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.GRU},  
    dtype=torch.qint8
)


torch.save(quantized_model, "model_oas\\best_model_quantized.pt")
print("[INFO] Quantized model saved as best_model_quantized.pt")
