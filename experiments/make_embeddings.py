"""Embed the AG News training set (120k news items) with all-MiniLM-L6-v2 (384-d).
Writes data/agnews_minilm.npy (float32). Runs on the Mac GPU in a few minutes."""
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

out = Path(__file__).resolve().parents[1] / "data" / "agnews_minilm.npy"
texts = load_dataset("fancyzhx/ag_news", split="train")["text"]
model = SentenceTransformer("all-MiniLM-L6-v2", device="mps")
emb = model.encode(texts, batch_size=256, show_progress_bar=False, convert_to_numpy=True)
np.save(out, emb.astype(np.float32))
print(emb.shape)
