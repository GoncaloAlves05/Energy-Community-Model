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
    "fixed"        : "Coef. Fixos",
    "proportional" : "Proporcional",
    "hierarchical" : "Hierárquico",
    "dynamic"      : "Dinâmico",
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
fig = plt.figure(figsize=(14, 10))
fig.suptitle(
    "Simulação de Comunidade de Energia Renovável — Resultados\n"
    f"N={N_MEMBERS} membros · 1 ano (8 760 horas) · Mesa 3.x · DL 15/2022 (PT)",
    fontsize=12, fontweight="bold", y=0.98
)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# ── Painel 1: Taxa de autoconsumo vs. % prosumers ──
for mode in SHARING_MODES:
    sub = df[df["sharing_mode"] == mode]
    ax1.plot(sub["prosumer_ratio"] * 100,
             sub["self_consumption_rate"] * 100,
             marker="o", color=COLORS[mode], label=LABELS[mode], linewidth=2)
ax1.set_title("P1+P2 · Taxa de Autoconsumo Coletivo")
ax1.set_xlabel("% Prosumers na comunidade")
ax1.set_ylabel("Taxa de autoconsumo (%)")
ax1.legend(fontsize=8)
ax1.set_xlim(5, 95)

# ── Painel 2: Taxa de autossuficiência vs. % prosumers ──
for mode in SHARING_MODES:
    sub = df[df["sharing_mode"] == mode]
    ax2.plot(sub["prosumer_ratio"] * 100,
             sub["self_sufficiency_rate"] * 100,
             marker="s", color=COLORS[mode], label=LABELS[mode], linewidth=2)
ax2.set_title("P1+P2 · Taxa de Autossuficiência")
ax2.set_xlabel("% Prosumers na comunidade")
ax2.set_ylabel("Taxa de autossuficiência (%)")
ax2.legend(fontsize=8)
ax2.set_xlim(5, 95)

# ── Painel 3: Poupança média por membro vs. % prosumers ──
for mode in SHARING_MODES:
    sub = df[df["sharing_mode"] == mode]
    ax3.plot(sub["prosumer_ratio"] * 100,
             sub["avg_savings_eur"],
             marker="^", color=COLORS[mode], label=LABELS[mode], linewidth=2)
ax3.set_title("P1+P2 · Poupança Média por Membro (€/ano)")
ax3.set_xlabel("% Prosumers na comunidade")
ax3.set_ylabel("Poupança média (€)")
ax3.legend(fontsize=8)
ax3.set_xlim(5, 95)

# ── Painel 4: Comparação dos modos para 50% prosumers ──
sub50  = df[df["prosumer_ratio"] == 0.5].copy()
sub50["label"] = sub50["sharing_mode"].map(LABELS)
x      = np.arange(len(sub50))
width  = 0.35

ax4.bar(x - width/2,
        sub50["self_consumption_rate"] * 100,
        width, label="Autoconsumo (%)",
        color=[COLORS[m] for m in sub50["sharing_mode"]], alpha=0.85)

ax4_r = ax4.twinx()
ax4_r.bar(x + width/2,
          sub50["avg_savings_eur"],
          width, label="Poupança (€)",
          color=[COLORS[m] for m in sub50["sharing_mode"]], alpha=0.4,
          hatch="//")

ax4.set_title("P2 · Modos de Partilha (50% prosumers)")
ax4.set_xlabel("Modo de partilha")
ax4.set_ylabel("Taxa de autoconsumo (%)")
ax4_r.set_ylabel("Poupança média (€/ano)")
ax4.set_xticks(x)
ax4.set_xticklabels(sub50["label"], rotation=12)

from matplotlib.patches import Patch
legend_els = [
    Patch(facecolor="gray", alpha=0.85, label="Autoconsumo (%)"),
    Patch(facecolor="gray", alpha=0.4, hatch="//", label="Poupança (€)"),
]
ax4.legend(handles=legend_els, fontsize=8, loc="upper left")

plt.savefig("../results.png", dpi=150, bbox_inches="tight")
print("Figura guardada: results.png")
plt.show()

