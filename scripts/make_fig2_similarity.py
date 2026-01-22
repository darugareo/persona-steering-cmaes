import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

CSV_PATH = Path("paper/tables/optimization_weights_10personas.csv")
OUT_PATH = Path("paper/figures/fig2_persona_similarity.png")

df = pd.read_csv(CSV_PATH)

persona_col = "persona" if "persona" in df.columns else df.columns[0]
personas = df[persona_col].astype(str).tolist()

wcols = [c for c in df.columns if c.lower() in ["w1","w2","w3","w4","w5","r1","r2","r3","r4","r5"]]
if len(wcols) < 5:
    num_df = df.select_dtypes(include=["number"])
    if num_df.shape[1] < 5:
        raise ValueError("Could not find 5 weight columns. Columns: " + str(df.columns))
    wcols = list(num_df.columns[-5:])

W = df[wcols].to_numpy(dtype=float)
sim = cosine_similarity(W)

plt.figure(figsize=(6.5, 5.5))
im = plt.imshow(sim, vmin=0, vmax=1, cmap='viridis')
plt.colorbar(im, fraction=0.046, pad=0.04)

plt.xticks(range(len(personas)), personas, rotation=45, ha="right", fontsize=8)
plt.yticks(range(len(personas)), personas, fontsize=8)

plt.xlabel("Persona")
plt.ylabel("Persona")
plt.title("Pairwise Similarity of Optimized Trait Weights")
plt.tight_layout()

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PATH, dpi=300)
plt.close()

print("saved:", OUT_PATH)
print("weights columns:", wcols)
