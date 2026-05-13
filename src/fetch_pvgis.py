"""
Fase 1 — Dados Reais de Produção Solar
=======================================
Integração com a PVGIS API (JRC / Comissão Europeia)
para obter perfis horários reais de produção solar.

Output: data/pvgis_porto_2023.csv
        data/solar_curve_normalised.npy  ← substitui solar_profile() no model.py

Utilização:
    python fetch_pvgis.py
    python fetch_pvgis.py --lat 38.72 --lon -9.14 --year 2022 --peak 5.0
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuração por defeito — Porto, 2023, 3 kWp (alinhado com model.py)
# ---------------------------------------------------------------------------
PVGIS_URL   = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
DEFAULT_LAT  = 41.15
DEFAULT_LON  = -8.61
DEFAULT_YEAR = 2020
DEFAULT_PEAK = 3.0    # kWp — valor de referência do model.py
DEFAULT_LOSS = 14     # % perdas do sistema (valor típico PVGIS para PT)

OUTPUT_DIR  = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# 1. Fetch da API
# ---------------------------------------------------------------------------

def fetch_pvgis(lat: float, lon: float, year: int, peak_kw: float,
                loss: float = DEFAULT_LOSS, retries: int = 3) -> dict:
    """
    Chama a PVGIS Hourly Radiation API e devolve o JSON completo.

    Parâmetros PVGIS relevantes:
        pvcalculation=1      → calcula produção PV (kW) em vez de só irradiação
        peakpower            → potência de pico do sistema (kWp)
        loss                 → perdas totais do sistema (%)
        hourlyoptimalangles=1 → usa inclinação e azimute óptimos para o local
        outputformat=json    → formato de resposta
    """
    params = {
    "lat"           : lat,
    "lon"           : lon,
    "startyear"     : year,
    "endyear"       : year,
    "pvcalculation" : 1,
    "peakpower"     : peak_kw,
    "loss"          : loss,
    "outputformat"  : "json",
    "angle"         : 30,
    "aspect"        : 0,
}

    print(f"A contactar PVGIS API...")
    print(f"  Local  : lat={lat}, lon={lon}")
    print(f"  Período: {year}")
    print(f"  Sistema: {peak_kw} kWp, perdas={loss}%")

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(PVGIS_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            print(f"  ✓ Resposta recebida com sucesso (tentativa {attempt})")
            return data
        except requests.exceptions.HTTPError as e:
            print(f"  ✗ HTTP {response.status_code}: {e}")
            raise
        except requests.exceptions.RequestException as e:
            print(f"  ✗ Tentativa {attempt}/{retries} falhou: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)  # backoff exponencial
            else:
                raise


# ---------------------------------------------------------------------------
# 2. Processar resposta → DataFrame horário
# ---------------------------------------------------------------------------

def parse_pvgis_response(data: dict, peak_kw: float) -> pd.DataFrame:
    """
    Converte o JSON da PVGIS num DataFrame com 8 760 linhas (1 por hora).

    Colunas produzidas:
        timestamp       : datetime UTC
        P_kW            : produção AC do sistema (kW) — valor bruto da PVGIS
        G_i_Wm2         : irradiação global incidente no plano do painel (W/m²)
        T_2m_C          : temperatura do ar a 2m (°C)  ← feature para ML
        solar_norm      : P_kW normalizado [0, 1] por peak_kw
    """
    hourly_raw = data["outputs"]["hourly"]
    n = len(hourly_raw)
    print(f"  Registos horários recebidos: {n}")

    # Verificação: a PVGIS pode devolver 8 760 ou 8 784 horas (ano bissexto)
    if n not in (8760, 8784):
        raise ValueError(f"Número inesperado de horas: {n}. Esperado 8760 ou 8784.")

    rows = []
    for rec in hourly_raw:
        rows.append({
            "timestamp" : rec["time"],          # ex: "20230101:0010"
            "P_kW"      : rec["P"] / 1000,      # PVGIS dá em W → converter para kW
            "G_i_Wm2"   : rec.get("G(i)", 0.0), # irradiação incidente (W/m²)
            "T_2m_C"    : rec.get("T2m", 0.0),  # temperatura (°C)
        })

    df = pd.DataFrame(rows)

    # Converter timestamp PVGIS ("20230101:0010") para datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%d:%H%M", utc=True)
    df = df.set_index("timestamp").sort_index()

    # Se ano bissexto (8784h), truncar para 8760h removendo 29 fev
    if len(df) == 8784:
        print("  Ano bissexto detectado — a remover 29 de Fevereiro (24h)")
        feb29 = df.index.month == 2
        mar1  = df.index.day == 29
        df    = df[~(feb29 & mar1)]
        print(f"  Registos após remoção: {len(df)}")

    assert len(df) == 8760, f"Esperado 8760 registos, obtido {len(df)}"

    # Normalizar produção [0, 1] pela potência de pico declarada
    # Usa peak_kw como denominador para manter consistência com model.py
    df["solar_norm"] = df["P_kW"] / peak_kw
    df["solar_norm"] = df["solar_norm"].clip(0.0, 1.0)  # garantia numérica

    return df.reset_index()


# ---------------------------------------------------------------------------
# 3. Guardar outputs
# ---------------------------------------------------------------------------

def save_outputs(df: pd.DataFrame, lat: float, lon: float,
                 year: int, peak_kw: float) -> None:
    """
    Guarda dois ficheiros em data/:
        pvgis_{location}_{year}.csv   → DataFrame completo (para análise e ML)
        solar_curve_normalised.npy    → array 1D de 8760 valores (para model.py)
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Nome descritivo do ficheiro
    location_tag = f"porto" if (abs(lat - 41.15) < 0.1) else f"{lat}_{lon}"
    csv_path     = OUTPUT_DIR / f"pvgis_{location_tag}_{year}.csv"
    npy_path     = OUTPUT_DIR / "solar_curve_normalised.npy"
    meta_path    = OUTPUT_DIR / "pvgis_metadata.json"

    # CSV completo — útil para feature engineering na Fase 2
    df.to_csv(csv_path, index=False)
    print(f"\n  ✓ CSV guardado: {csv_path}")

    # Array numpy — substituto directo de solar_profile() no model.py
    solar_array = df["solar_norm"].to_numpy(dtype=np.float32)
    np.save(npy_path, solar_array)
    print(f"  ✓ Array numpy guardado: {npy_path}  shape={solar_array.shape}")

    # Metadata — rastreabilidade (importante para o README e post técnico)
    meta = {
        "source"       : "PVGIS v5.2 — JRC / European Commission",
        "url"          : PVGIS_URL,
        "lat"          : lat,
        "lon"          : lon,
        "year"         : year,
        "peak_kw"      : peak_kw,
        "loss_pct"     : DEFAULT_LOSS,
        "n_hours"      : len(df),
        "P_kW_max"     : round(float(df["P_kW"].max()), 4),
        "P_kW_mean"    : round(float(df["P_kW"].mean()), 4),
        "solar_norm_max" : round(float(solar_array.max()), 4),
        "annual_yield_kWh": round(float(df["P_kW"].sum()), 1),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  ✓ Metadata guardada: {meta_path}")

    # Sumário legível
    print(f"\n{'─'*55}")
    print(f"  Produção anual estimada : {meta['annual_yield_kWh']:.0f} kWh")
    print(f"  Pico de produção        : {meta['P_kW_max']:.3f} kW")
    print(f"  Produção média horária  : {meta['P_kW_mean']:.4f} kW")
    print(f"{'─'*55}")


# ---------------------------------------------------------------------------
# 4. Validação rápida — sanity checks
# ---------------------------------------------------------------------------

def validate(df: pd.DataFrame) -> None:
    """
    Verifica propriedades básicas do perfil solar obtido.
    Falha com AssertionError se algo estiver claramente errado.
    """
    arr = df["solar_norm"].to_numpy()

    assert len(arr) == 8760,         f"Esperado 8760 horas, obtido {len(arr)}"
    assert arr.min() >= 0.0,         f"Valores negativos encontrados: {arr.min()}"
    assert arr.max() <= 1.0,         f"Normalização falhou: max={arr.max()}"
    assert arr.max() > 0.5,          f"Pico solar suspeito: max={arr.max():.3f}"

    # Verificar que existem zeros nocturnos (deve ser >40% das horas)
    night_frac = (arr == 0.0).mean()
    assert night_frac > 0.40, f"Fracção nocturna suspeita: {night_frac:.1%}"

    # Verificar sazonalidade: verão (horas 2880-5040) > inverno (horas 0-720)
    summer_mean = arr[2880:5040].mean()
    winter_mean = arr[0:720].mean()
    assert summer_mean > winter_mean, (
        f"Sazonalidade invertida: verão={summer_mean:.4f} < inverno={winter_mean:.4f}"
    )

    print(f"\n  ✓ Validação passou")
    print(f"    Horas com produção > 0 : {(arr > 0).sum()} ({(arr > 0).mean():.1%})")
    print(f"    Média verão vs inverno : {summer_mean:.4f} vs {winter_mean:.4f}")


# ---------------------------------------------------------------------------
# 5. Integração com model.py — como usar o array gerado
# ---------------------------------------------------------------------------

INTEGRATION_HINT = """
─────────────────────────────────────────────────────────
Como integrar no model.py (Fase 2):

No Household.__init__, substituir:
    self._solar_curve = solar_profile(8760, agent_rng) if is_prosumer else np.zeros(8760)

Por:
    base_curve = np.load("data/solar_curve_normalised.npy")
    noise = agent_rng.uniform(0.92, 1.08, size=8760)  # variabilidade individual ±8%
    self._solar_curve = np.clip(base_curve * noise, 0, 1) if is_prosumer else np.zeros(8760)

Isto mantém a heterogeneidade entre agentes (cada prosumer tem
um perfil ligeiramente diferente) usando dados reais como base.
─────────────────────────────────────────────────────────
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch dados horários de produção solar — PVGIS API"
    )
    parser.add_argument("--lat",  type=float, default=DEFAULT_LAT,
                        help=f"Latitude (default: {DEFAULT_LAT} — Porto)")
    parser.add_argument("--lon",  type=float, default=DEFAULT_LON,
                        help=f"Longitude (default: {DEFAULT_LON} — Porto)")
    parser.add_argument("--year", type=int,   default=DEFAULT_YEAR,
                        help=f"Ano de referência (default: {DEFAULT_YEAR})")
    parser.add_argument("--peak", type=float, default=DEFAULT_PEAK,
                        help=f"Potência de pico kWp (default: {DEFAULT_PEAK})")
    args = parser.parse_args()

    print("=" * 55)
    print("  PVGIS Fetch — CER Porto")
    print("=" * 55)

    # 1. Fetch
    raw_data = fetch_pvgis(args.lat, args.lon, args.year, args.peak)

    # 2. Processar
    df = parse_pvgis_response(raw_data, args.peak)

    # 3. Validar
    validate(df)

    # 4. Guardar
    save_outputs(df, args.lat, args.lon, args.year, args.peak)

    # 5. Hint de integração
    print(INTEGRATION_HINT)


if __name__ == "__main__":
    main()
