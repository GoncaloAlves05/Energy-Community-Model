"""
Análise da Comunidade de Energia — P1 e P2
===========================================
P1: Efeito da composição (% prosumers) nos outcomes coletivos
P2: Comparação dos 4 modos de partilha (RAC — Regulamento de Acesso às Redes)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from model import EnergyCommunity, SHARING_MODES

# ---------------------------------------------------------------------------
# Configuração de estilo dos gráficos
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor"  : "#f8f9fa",
    "axes.grid"       : True,
    "grid.color"      : "white",
    "grid.linewidth"  : 1.2,
    "font.family"     : "sans-serif",
    "font.size"       : 10,
    "axes.titlesize"  : 11,
    "axes.titleweight": "bold",
    "axes.labelsize"  : 10,
})

# Cores e etiquetas para cada modo de partilha
COLORS = {
    "fixed"        : "#2196F3",
    "proportional" : "#4CAF50",
    "hierarchical" : "#FF9800",
    "dynamic"      : "#9C27B0",
}
LABELS = {
    "fixed"        : "Fixed Coef.",
    "proportional" : "Proportional",
    "hierarchical" : "Hierarchical",
    "dynamic"      : "Dynamic",
}

# ---------------------------------------------------------------------------
# Parâmetros do batch de simulações
# ---------------------------------------------------------------------------
PROSUMER_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_MEMBERS       = 30
N_HOURS         = 8760   # 1 ano completo
SEED            = 42

# ---------------------------------------------------------------------------
# Correr simulações
# ---------------------------------------------------------------------------
print("A correr simulações...")
results = []
total   = len(PROSUMER_RATIOS) * len(SHARING_MODES)
done    = 0

for ratio in PROSUMER_RATIOS:
    for mode in SHARING_MODES:
        model  = EnergyCommunity(
            n_members      = N_MEMBERS,
            prosumer_ratio = ratio,
            sharing_mode   = mode,
            n_hours        = N_HOURS,
            rng            = SEED,
        )
        summary = model.run()
        results.append(summary)
        done += 1
        print(f"  [{done}/{total}] prosumers={ratio:.0%}  "
              f"modo={mode:>12s}  "
              f"autoconsumo={summary['self_consumption_rate']:.1%}  "
              f"poupança={summary['avg_savings_eur']:.0f}€")

df = pd.DataFrame(results)
print("\nSimulações concluídas.\n")

# ---------------------------------------------------------------------------
# Figura principal — 4 painéis
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 11))
fig.patch.set_facecolor("#0f1117")
fig.suptitle(
    "Simulating Renewable Energy Communities with Agent-Based Modelling",
    fontsize=15, fontweight="bold", y=0.98, color="white"
)
fig.text(0.5, 0.94,
         "30 members · 1 year (8,760 hourly ticks) · Mesa 3.x · DL 15/2022 (Portugal)",
         ha="center", fontsize=10, color="#aaaaaa")

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.35)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

DARK_BG   = "#1e2130"
TEXT_COLOR = "white"
GRID_COLOR = "#2e3250"

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2e3250")
    ax.grid(color=GRID_COLOR, linewidth=0.4, zorder=0)

# ── Painel 1: Self-consumption rate ──
for mode in SHARING_MODES:
    sub = df[df["sharing_mode"] == mode]
    ax1.plot(sub["prosumer_ratio"] * 100,
             sub["self_consumption_rate"] * 100,
             marker="o", color=COLORS[mode], label=LABELS[mode], linewidth=2.5)
ax1.set_title("Collective Self-Consumption Rate")
ax1.set_xlabel("% Prosumers in the community")
ax1.set_ylabel("Self-consumption rate (%)")
ax1.legend(fontsize=8, facecolor=DARK_BG, labelcolor=TEXT_COLOR)
ax1.set_xlim(5, 95)

# ── Painel 2: Self-sufficiency rate ──
for mode in SHARING_MODES:
    sub = df[df["sharing_mode"] == mode]
    ax2.plot(sub["prosumer_ratio"] * 100,
             sub["self_sufficiency_rate"] * 100,
             marker="s", color=COLORS[mode], label=LABELS[mode], linewidth=2.5)
ax2.set_title("Community Self-Sufficiency Rate")
ax2.set_xlabel("% Prosumers in the community")
ax2.set_ylabel("Self-sufficiency rate (%)")
ax2.legend(fontsize=8, facecolor=DARK_BG, labelcolor=TEXT_COLOR)
ax2.set_xlim(5, 95)

# ── Painel 3: Average savings ──
for mode in SHARING_MODES:
    sub = df[df["sharing_mode"] == mode]
    ax3.plot(sub["prosumer_ratio"] * 100,
             sub["avg_savings_eur"],
             marker="^", color=COLORS[mode], label=LABELS[mode], linewidth=2.5)
ax3.set_title("Average Savings per Member (€/year)")
ax3.set_xlabel("% Prosumers in the community")
ax3.set_ylabel("Average savings (€)")
ax3.legend(fontsize=8, facecolor=DARK_BG, labelcolor=TEXT_COLOR)
ax3.set_xlim(5, 95)

# ── Painel 4: Sharing modes at 50% prosumers ──
sub50  = df[df["prosumer_ratio"] == 0.5].copy()
sub50["label"] = sub50["sharing_mode"].map(LABELS)
x      = np.arange(len(sub50))
width  = 0.35

ax4.bar(x - width/2,
        sub50["self_consumption_rate"] * 100,
        width, label="Self-consumption (%)",
        color=[COLORS[m] for m in sub50["sharing_mode"]], alpha=0.9)

ax4_r = ax4.twinx()
ax4_r.set_facecolor(DARK_BG)
ax4_r.grid(False)
ax4_r.tick_params(colors=TEXT_COLOR)
ax4_r.yaxis.label.set_color(TEXT_COLOR)
for spine in ax4_r.spines.values():
    spine.set_edgecolor(GRID_COLOR)

ax4_r.bar(x + width/2,
          sub50["avg_savings_eur"],
          width, label="Avg. savings (€)",
          color=[COLORS[m] for m in sub50["sharing_mode"]], alpha=0.4,
          hatch="//")

ax4.set_title("Sharing Modes Compared (50% prosumers)")
ax4.set_xlabel("Sharing mode")
ax4.set_ylabel("Self-consumption rate (%)", color=TEXT_COLOR)
ax4_r.set_ylabel("Average savings (€/year)", color=TEXT_COLOR)
ax4.set_xticks(x)
ax4.set_xticklabels(sub50["label"], rotation=12, color=TEXT_COLOR)

from matplotlib.patches import Patch
legend_els = [
    Patch(facecolor="gray", alpha=0.9, label="Self-consumption (%)"),
    Patch(facecolor="gray", alpha=0.4, hatch="//", label="Avg. savings (€)"),
]
ax4.legend(handles=legend_els, fontsize=8, loc="upper left",
           facecolor=DARK_BG, labelcolor=TEXT_COLOR)

plt.savefig("../results.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Figure saved: results.png")
plt.show()