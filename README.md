# Energy Community Simulation 🌱

Agent-Based Model (ABM) of a Renewable Energy Community (REC) in Portugal,
built with [Mesa 3.x](https://mesa.readthedocs.io/) in Python.

## Motivation

Renewable Energy Communities (RECs) are a key instrument of Portugal's energy
transition, formalized by DL 15/2022. Despite their potential, fewer than 400
collective self-consumption units exist in Portugal today. This model simulates
how community composition and energy sharing rules affect collective outcomes.

## Research Questions

- **P1.** How does community composition (% of prosumers) affect collective
  self-consumption and average savings?
- **P2.** How do the 4 sharing modes defined by ERSE (RAC nº 815/2023) compare
  in terms of efficiency and economic benefit?

## Model Overview

| Component | Description |
|---|---|
| `Household` | Member agent — consumes energy, produces solar if prosumer |
| `EGAC` | Community manager — distributes energy each tick, tracks metrics |
| `EnergyCommunity` | Mesa model — 30 members, 8 760 ticks (1 year, hourly) |

**Sharing modes simulated:** Fixed coefficients · Proportional · Hierarchical · Dynamic

## Key Results

![Results](results.png)

- Collective self-consumption **decreases** as prosumer share grows — excess
  production finds no internal demand and is exported to the grid at low prices
- Self-sufficiency peaks around **50% prosumers** — the community's optimal balance
- The **Dynamic sharing mode** consistently outperforms others in both
  self-consumption rate and average savings per member
- With 50% prosumers and dynamic sharing, members save up to **~250€/year**
  without any battery storage

## Regulatory Context

| Reference | Description |
|---|---|
| DL 15/2022 | Legal framework for RECs in Portugal |
| RAC nº 815/2023 (ERSE) | Defines the 4 energy sharing modes simulated |
| PVGIS (EU Commission) | Solar irradiation data reference for Portugal |
| ERSE BTN profiles | Hourly consumption profiles (residential & small business) |

