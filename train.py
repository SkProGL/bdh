# Copyright Pathway Technology, Inc.

import torch.nn.functional as F
import torch.nn as nn
import torch
import requests
import bdh
from contextlib import nullcontext
import os
import numpy as np
import pandas as pd  # <<< REQUIRED CHANGE


# =========================
# Latent visualization (UNCHANGED)
# =========================


def inspect_next_token_logits(model, idx, top_k=10):
    model.eval()
    with torch.no_grad():
        logits, _ = model(idx)
        next_logits = logits[:, -1, :]
        values, indices = torch.topk(next_logits, top_k)

    tokens = [chr(i) for i in indices[0].tolist()]
    scores = values[0].tolist()

    print("\nNext-token possibilities:")
    for rank, (tok, score) in enumerate(zip(tokens, scores), 1):
        printable = tok if tok.isprintable() else repr(tok)
        print(f"{rank:>2}. '{printable}'  logit={score:.3f}")


# =========================
# Device & dtype (UNCHANGED)
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)

ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]

ctx = (
    torch.amp.autocast(device_type=device.type, dtype=ptdtype)
    if "cuda" in device.type
    else nullcontext()
)

scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype == "float16"))

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print(f"Using device: {device} with dtype {dtype}")


# =========================
# Configuration (UNCHANGED)
# =========================

BDH_CONFIG = bdh.BDHConfig()
BLOCK_SIZE = 512
BATCH_SIZE = 4
MAX_ITERS = 1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.1
LOG_FREQ = 100

# =========================
# Data: wiki.parquet (REQUIRED CHANGE)
# =========================

WIKI_PATH = os.path.join(os.path.dirname(__file__), "wiki.parquet")
_wiki_bytes = None


def fetch_data():
    assert os.path.exists(WIKI_PATH), "wiki.parquet not found"


def load_wiki_bytes():
    global _wiki_bytes
    if _wiki_bytes is not None:
        return _wiki_bytes

    df = pd.read_parquet(WIKI_PATH)

    texts = []
    for _, row in df.iterrows():
        title = str(row.iloc[0])
        body = str(row.iloc[1])
        texts.append(f"{title}\n{body}")

    full_text = "\n\n<doc>\n\n".join(texts)

    _wiki_bytes = np.frombuffer(
        full_text.encode("utf-8"),
        dtype=np.uint8,
    )

    return _wiki_bytes


def get_batch(split):
    data = load_wiki_bytes()

    split_idx = int(0.9 * len(data))
    data = data[:split_idx] if split == "train" else data[split_idx:]

    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))

    x = torch.stack(
        [torch.from_numpy(data[i: i + BLOCK_SIZE].astype(np.int64))
         for i in ix]
    )
    y = torch.stack(
        [torch.from_numpy(data[i + 1: i + 1 + BLOCK_SIZE].astype(np.int64))
         for i in ix]
    )

    if torch.cuda.is_available():
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y


def save_model(raw_model):
    SAVE_PATH = "bdh_counting.pt"

    checkpoint = {
        "model_state_dict": raw_model.state_dict(),
        "config": vars(BDH_CONFIG),
    }

    torch.save(checkpoint, SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")


# =========================
# Training
# =========================
if __name__ == "__main__":
    fetch_data()

    raw_model = bdh.BDH(BDH_CONFIG).to(device)
    model = torch.compile(raw_model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    x, y = get_batch("train")

    for step in range(MAX_ITERS):
        with ctx:
            logits, loss = model(x, y)

        x, y = get_batch("train")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if step % LOG_FREQ == 0:
            print(f"Step {step}/{MAX_ITERS} loss {loss.item():.4f}")

    save_model(raw_model)
    print("Training done, now generating a sample")

    model.eval()
    prompt = torch.tensor(
        bytearray("To be or ", "utf-8"),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    inspect_next_token_logits(model, prompt, top_k=10)

    ret = model.generate(prompt, max_new_tokens=100, top_k=3)
    ret_decoded = bytes(ret.to(torch.uint8).cpu().squeeze(0)).decode(
        errors="backslashreplace"
    )
    print(ret_decoded)

    # =========================
    # Latent visualization (UNCHANGED)
    # =========================

    with torch.no_grad():
        _, _, x_sparse = model(prompt, return_latents=True)

    x_vis = x_sparse.mean(dim=(0, 1)).cpu().numpy()
    tokens = [chr(c) for c in prompt[0].cpu().tolist()]
