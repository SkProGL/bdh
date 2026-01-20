# Copyright Pathway Technology, Inc.

import os
from contextlib import nullcontext

import bdh
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# On a Mac you can also try
# device=torch.device('mps')

dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
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
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
print(f"Using device: {device} with dtype {dtype}")


# Configuration
BDH_CONFIG = bdh.BDHConfig()
# BLOCK_SIZE = 512
# BATCH_SIZE = 32

BLOCK_SIZE = 32
BATCH_SIZE = 64
MAX_ITERS = 3000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
LOG_FREQ = 100

input_file_path = os.path.join(os.path.dirname(__file__), "count.txt")


def fetch_data():
    assert os.path.exists(input_file_path), "count.txt not found"


def get_batch(split):
    with open(input_file_path, "r") as f:
        lines = f.read().splitlines()

    split_idx = int(0.9 * len(lines))
    if split == "train":
        lines = lines[:split_idx]
    else:
        lines = lines[split_idx:]

    ix = torch.randint(len(lines), (BATCH_SIZE,))
    batch = [lines[i] for i in ix]

    PAD = 32  # ASCII space

    encoded = []
    for s in batch:
        e = torch.tensor(bytearray(s, "utf-8"), dtype=torch.long)

        if len(e) < BLOCK_SIZE + 1:
            pad_len = BLOCK_SIZE + 1 - len(e)
            e = torch.cat([e, torch.full((pad_len,), PAD, dtype=torch.long)])
        else:
            e = e[:BLOCK_SIZE + 1]

        encoded.append(e)

    encoded = torch.stack(encoded)  # (B, BLOCK_SIZE+1)

    x = encoded[:, :BLOCK_SIZE]
    y = encoded[:, 1:BLOCK_SIZE+1]

    if torch.cuda.is_available():
        x, y = x.pin_memory().to(device, non_blocking=True), \
            y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y
# def get_batch(split):
#     # treat the file as bytes
#     data = np.memmap(input_file_path, dtype=np.uint8, mode="r")
#     if split == "train":
#         data = data[: int(0.9 * len(data))]
#     else:
#         data = data[int(0.9 * len(data)):]
#     ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
#     x = torch.stack(
#         [torch.from_numpy((data[i: i + BLOCK_SIZE]).astype(np.int64))
#          for i in ix]
#     )
#     y = torch.stack(
#         [
#             torch.from_numpy(
#                 (data[i + 1: i + 1 + BLOCK_SIZE]).astype(np.int64))
#             for i in ix
#         ]
#     )
#     if torch.cuda.is_available():
#         # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
#         x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
#             device, non_blocking=True
#         )
#     else:
#         x, y = x.to(device), y.to(device)
#     return x, y


def eval(model):
    model.eval()


if __name__ == "__main__":
    fetch_data()

    model = bdh.BDH(BDH_CONFIG).to(device)
    model = torch.compile(model)
    # model = torch.compile(model, backend="eager")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    x, y = get_batch("train")

    loss_acc = 0
    loss_steps = 0
    for step in range(MAX_ITERS):
        with ctx:
            logits, loss = model(x, y)
        x, y = get_batch("train")
        loss_acc += loss
        loss_steps += 1
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if step % LOG_FREQ == 0:
            print(
                f"Step: {step}/{MAX_ITERS} loss {loss_acc.item() / loss_steps:.3}")
            loss_acc = 0
            loss_steps = 0
    print("Training done, now generating a sample ")
    model.eval()
    prompt = torch.tensor(
        bytearray("0 ", "utf-8"), dtype=torch.long, device=device
    ).unsqueeze(0)
    # ret = model.generate(prompt, max_new_tokens=100, top_k=3)
    ret = model.generate(
        prompt,
        max_new_tokens=1000,
        top_k=1,
        temperature=1.0,
    )
    ret_decoded = bytes(ret.to(torch.uint8).to("cpu").squeeze(0)).decode(
        errors="backslashreplace"
    )
    print(ret_decoded)
