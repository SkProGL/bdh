import torch
import bdh

# -------- device & dtype --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    dtype = (
        "bfloat16"
        if torch.cuda.is_bf16_supported()
        else "float16"
    )
else:
    dtype = "float32"

ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]

# -------- load checkpoint --------
CKPT_PATH = "bdh_counting.pt"

checkpoint = torch.load(CKPT_PATH, map_location=device)

config = bdh.BDHConfig(**checkpoint["config"])
model = bdh.BDH(config).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Optional: compile for faster inference
model = torch.compile(model)

# -------- inference --------
prompt = torch.tensor(
    bytearray("0 ", "utf-8"),
    dtype=torch.long,
    device=device,
).unsqueeze(0)

with torch.no_grad():
    with torch.autocast(
        device_type=device.type,
        dtype=ptdtype,
        enabled=(device.type == "cuda"),
    ):
        out = model.generate(
            prompt,
            max_new_tokens=1000,
            top_k=1,
            temperature=1.0,
        )

decoded = bytes(
    out.to(torch.uint8).cpu().squeeze(0)
).decode(errors="replace")

print(decoded)
