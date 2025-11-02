# scripts/plot_th_sweep.py
import pandas as pd, matplotlib.pyplot as plt
p="outputs/cvc_eval_legacy_raw/th_sweep.tsv"
df=pd.read_csv(p, sep="\t")
plt.figure()
plt.plot(df["th"], df["dice"], marker="o")
plt.xlabel("Threshold"); plt.ylabel("Dice"); plt.title("CVC-ClinicDB Threshold Sweep")
plt.grid(True)
plt.savefig(p.replace("th_sweep.tsv","th_sweep_dice.png"), dpi=200, bbox_inches="tight")
print("saved plot.")
