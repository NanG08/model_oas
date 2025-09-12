import torch
from torch.serialization import add_safe_globals
from action_model import CNN_GRU   # import your model definition

# allow unpickling of CNN_GRU
add_safe_globals([CNN_GRU, torch.ScriptObject])

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "model_oas\\best_model.pt"

checkpoint = torch.load(model_path, map_location=device, weights_only=False)

# If you saved the full model:
if isinstance(checkpoint, torch.nn.Module):
    model = checkpoint
# If you saved only the state_dict:
else:
    model = CNN_GRU(num_classes=5, hidden_size=128)
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

# Now you can export to ONNX
dummy_input = torch.randn(1, 16, 3, 112, 112).to(device)  # adjust seq_len/size as needed
torch.onnx.export(
    model,
    dummy_input,
    "action_model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=12
)
print("[INFO] Exported model to ONNX successfully")
