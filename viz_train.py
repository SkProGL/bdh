
import torch.nn.functional as F
import torch.nn as nn
import torch
import requests
import bdh
from contextlib import nullcontext
import os
import numpy as np


def launch_bdh_latent_grid(
    x_sparse,          # np.ndarray, shape (T, D) or (T, ...)
    tokens,            # list[str], length T
    *,
    colorscale="BuGn",
    columns=2,
    height=2300,
    debug=True,
):
    """
    Launch a Dash app showing per-token latent feature activations
    laid out by PCA over latent feature activity.

    Parameters
    ----------
    x_sparse : np.ndarray
        Latent activations per token. Expected shape (T, D).
        (If higher-rank, caller should reduce beforehand.)
    tokens : list[str]
        Token strings corresponding to each timestep.
    colorscale : str
        Plotly colorscale name.
    columns : int
        Number of grid columns.
    height : int
        Height of each subplot in pixels.
    debug : bool
        Dash debug flag.
    """

    import numpy as np
    import plotly.graph_objects as go
    from sklearn.decomposition import PCA
    from dash import Dash, html, dcc

    x_sparse = np.asarray(x_sparse)
    T, D = x_sparse.shape

    assert len(tokens) == T, "tokens length must match x_sparse first dimension"

    # ----------------------------
    # Normalize activations (for color)
    # ----------------------------

    x_norm = x_sparse / (np.max(x_sparse) + 1e-8)

    # ----------------------------
    # Grid layout over latent features (CLEAN & HONEST)
    # ----------------------------

    cols = int(np.ceil(np.sqrt(D)))
    rows = int(np.ceil(D / cols))

    x_pos = np.array([i % cols for i in range(D)])
    y_pos = np.array([-(i // cols) for i in range(D)])  # negative for top-down

    # ----------------------------
    # Per-token figure factory
    # ----------------------------

    def make_figure(t, token):
        return go.Figure(
            data=[
                go.Scatter(
                    x=x_pos,
                    y=y_pos,
                    # mode="markers+text",
                    mode="markers",
                    text=[str(i) for i in range(D)],
                    textposition="middle center",
                    marker=dict(
                        size=14,
                        color=x_norm[t],
                        colorscale=colorscale,
                        cmin=0.0,
                        cmax=1.0,
                        line=dict(width=1, color="black"),
                        showscale=True,
                        colorbar=dict(
                            title="Activation",
                            len=0.7,
                        ),
                    ),
                    hovertemplate=(
                        "Latent feature %{text}<br>"
                        "Activation %{marker.color:.3f}<extra></extra>"
                    ),
                )
            ],
            layout=go.Layout(
                title=f'Token {t}: "{token}"',
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                template="plotly_white",
                height=height,
                margin=dict(l=10, r=10, t=30, b=10),
            ),
        )

    # ----------------------------
    # Dash app
    # ----------------------------

    app = Dash(__name__)

    grid_items = [
        html.Div(
            dcc.Graph(
                figure=make_figure(t, token),
                config={"displayModeBar": False},
            ),
            style={
                "border": "1px solid #ddd",
                "borderRadius": "6px",
                "padding": "4px",
                "backgroundColor": "white",
            },
        )
        for t, token in enumerate(tokens)
    ]

    app.layout = html.Div(
        [
            html.H3("BDH Latent Feature Activation per Token"),
            html.P(
                "Each panel shows latent feature activations for one token. "
                "Feature positions are fixed via PCA over activation patterns."
            ),
            html.Div(
                grid_items,
                style={
                    "display": "grid",
                    # "gridTemplateColumns": " ".join(["1fr"] * columns),
                    "gridTemplateColumns": "repeat(auto-fill, minmax(500px, 1fr))",
                    "gap": "12px",
                },
            ),
        ],
        style={
            "margin": "0 auto",
            "padding": "16px",
            "fontFamily": "sans-serif",
        },
    )
    figs = [make_figure(t, token) for t, token in enumerate(tokens)]

    import plotly.io as pio

    for t, token in enumerate(tokens):
        fig = make_figure(t, token)
        pio.write_html(
            fig,
            file=f"bdh_latent_{t}_{token}.html",
            include_plotlyjs="cdn",
            auto_open=(t == 0),
        )

    return
    app.run(debug=False)


def inspect_next_token_logits(model, idx, top_k=10):
    """
    Inspect top-k next-token possibilities for a given prompt.
    """
    model.eval()
    with torch.no_grad():
        logits, _ = model(idx)
        next_logits = logits[:, -1, :]  # (1, vocab)
        values, indices = torch.topk(next_logits, top_k)

    tokens = [chr(i) for i in indices[0].tolist()]  # byte-level vocab
    scores = values[0].tolist()

    print("\nNext-token possibilities:")
    for rank, (tok, score) in enumerate(zip(tokens, scores), 1):
        printable = tok if tok.isprintable() else repr(tok)
        print(f"{rank:>2}. '{printable}'  logit={score:.3f}")

# Copyright Pathway Technology, Inc.


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
BLOCK_SIZE = 512
# BATCH_SIZE = 32
# MAX_ITERS = 3000
BATCH_SIZE = 4
MAX_ITERS = 1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.1
LOG_FREQ = 100

input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")


# Fetch the tiny Shakespeare dataset
def fetch_data():
    if not os.path.exists(input_file_path):
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_file_path, "w") as f:
            f.write(requests.get(data_url).text)


def get_batch(split):
    # treat the file as bytes
    data = np.memmap(input_file_path, dtype=np.uint8, mode="r")
    if split == "train":
        data = data[: int(0.9 * len(data))]
    else:
        data = data[int(0.9 * len(data)):]
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack(
        [torch.from_numpy((data[i: i + BLOCK_SIZE]).astype(np.int64))
         for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy(
                (data[i + 1: i + 1 + BLOCK_SIZE]).astype(np.int64))
            for i in ix
        ]
    )
    if torch.cuda.is_available():
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def eval(model):
    model.eval()


if __name__ == "__main__":
    fetch_data()

    model = bdh.BDH(BDH_CONFIG).to(device)
    model = torch.compile(model)
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
        bytearray("To be or ", "utf-8"), dtype=torch.long, device=device
    ).unsqueeze(0)

    # Inspect possibilities BEFORE sampling
    inspect_next_token_logits(model, prompt, top_k=10)

    ret = model.generate(prompt, max_new_tokens=100, top_k=3)
    ret_decoded = bytes(ret.to(torch.uint8).to("cpu").squeeze(0)).decode(
        errors="backslashreplace"
    )
    print(ret_decoded)
    # ----------------------------
    # Visualize BDH latent states (MINIMAL)
    # ----------------------------

    model.eval()
    with torch.no_grad():
        # run a clean forward pass on the prompt
        logits, loss = model(prompt)

        # Re-run forward manually to capture x_sparse
        # (we rely on the last-layer x_sparse via a hook-like pattern)
        _, _, x_sparse = model(prompt, return_latents=True)

    # x_sparse shape: (B, nh, T, N) → reduce to (T, D)
    x_vis = x_sparse.mean(dim=(0, 1)).cpu().numpy()

    # prompt tokens as characters (since this is byte-level)
    tokens = [chr(c) for c in prompt[0].cpu().tolist()]

    launch_bdh_latent_grid(
        x_sparse=x_vis,
        tokens=tokens,
        columns=2,
    )
