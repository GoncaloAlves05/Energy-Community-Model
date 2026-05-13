"""
Fase 1 — Perfis de Consumo ERSE
================================
Processa o ficheiro oficial da E-REDES com perfis horários BTN
e produz arrays normalizados para uso no modelo Mesa.

Input : data/raw/E-REDES_Perfil_Consumo_2023.xlsx
Output: data/consumption_curve_btn_a.npy  ← residencial (substitui 'residential')
        data/consumption_curve_btn_c.npy  ← pequenos negócios (substitui 'small_business')
        data/erse_hourly_2023.csv          ← DataFrame horário completo (para ML na Fase 2)
        data/erse_metadata.json            ← rastreabilidade da fonte

Utilização:
    python process_erse_profiles.py
    python process_erse_profiles.py --input data/raw/E-REDES_Perfil_Consumo_2023.xlsx
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
DEFAULT_INPUT = Path("data/raw/E-REDES_Perfil_Consumo_2023.xlsx")
OUTPUT_DIR    = Path("data")

# Mapeamento entre perfis ERSE e tipos de agente do model.py
PROFILE_MAP = {
    "BTN_A": "residential",      # Perfil A — residencial sem aquecimento eléctrico
    "BTN_C": "small_business",   # Perfil C — pequenos negócios / comércio
}


# ---------------------------------------------------------------------------
# 1. Ler e limpar o ficheiro Excel
# ---------------------------------------------------------------------------

def load_erse_excel(path: Path) -> pd.DataFrame:
    """
    Lê o ficheiro E-REDES e devolve um DataFrame limpo com granularidade quarto-horária.

    Estrutura do ficheiro:
        - Linha 0: cabeçalho geral ("Consumo")
        - Linha 1: vazia
        - Linha 2: nomes dos perfis (BTN A, BTN B, BTN C, IP)
        - Linhas 3+: dados — 35 040 períodos de 15 min (365 dias × 96 períodos)
    """
    print(f"A ler ficheiro: {path}")

    df = pd.read_excel(path, skiprows=2, header=0)
    df.columns = ["Data", "Dia", "Hora", "BTN_A", "BTN_B", "BTN_C", "IP"]

    # Remover linha de cabeçalho residual e linhas sem dados
    df = df[df["BTN_A"] != "BTN A"].copy()
    df = df.dropna(subset=["BTN_A"]).reset_index(drop=True)

    # Converter colunas numéricas
    for col in ["BTN_A", "BTN_B", "BTN_C", "IP"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Construir timestamp completo a partir de Data + Hora
    # A coluna Hora está em formato "HH:MM" (string)
    # Nota: o ficheiro ERSE usa "24:00" para o último período do dia
    #       → tratar como 00:00 do dia seguinte
    df["Data_str"] = pd.to_datetime(df["Data"]).dt.strftime("%Y-%m-%d")
    df["Hora_fix"] = df["Hora"].astype(str)

    mask_24 = df["Hora_fix"] == "24:00"
    df.loc[mask_24, "Hora_fix"] = "00:00"
    df.loc[mask_24, "Data_str"] = (
        pd.to_datetime(df.loc[mask_24, "Data_str"]) + pd.Timedelta(days=1)
    ).dt.strftime("%Y-%m-%d")

    df["timestamp_qh"] = pd.to_datetime(
        df["Data_str"] + " " + df["Hora_fix"],
        format="%Y-%m-%d %H:%M",
        errors="coerce"
    )
    df = df.dropna(subset=["timestamp_qh"])
    df = df.sort_values("timestamp_qh").reset_index(drop=True)

    n = len(df)
    print(f"  Períodos quarto-horários carregados: {n}")
    if n not in (35040, 35136):  # 35136 = ano bissexto
        print(f"  ⚠ Número inesperado de períodos: {n} (esperado 35040)")

    return df[["timestamp_qh", "BTN_A", "BTN_B", "BTN_C", "IP"]]


# ---------------------------------------------------------------------------
# 2. Agregar de 15 min para 1 hora
# ---------------------------------------------------------------------------

def aggregate_to_hourly(df_qh: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega os 35 040 períodos quarto-horários em 8 760 horas.

    Os perfis ERSE são adimensionais e normalizados (soma anual = 1).
    A agregação soma os 4 períodos de cada hora — mantém a propriedade
    de normalização (soma anual continua = 1 após agregação).
    """
    df = df_qh.copy()
    df["hour"] = df["timestamp_qh"].dt.floor("h")

    df_hourly = (
        df.groupby("hour")[["BTN_A", "BTN_B", "BTN_C", "IP"]]
        .sum()
        .reset_index()
        .rename(columns={"hour": "timestamp"})
    )

    n = len(df_hourly)
    print(f"  Horas após agregação: {n}")

    # Se ano bissexto, remover 29 de Fevereiro
    if n == 8784:
        print("  Ano bissexto detectado — a remover 29 de Fevereiro")
        mask = ~((df_hourly["timestamp"].dt.month == 2) &
                 (df_hourly["timestamp"].dt.day == 29))
        df_hourly = df_hourly[mask].reset_index(drop=True)
        print(f"  Horas após remoção: {len(df_hourly)}")

    assert len(df_hourly) == 8760, f"Esperado 8760 horas, obtido {len(df_hourly)}"
    return df_hourly


# ---------------------------------------------------------------------------
# 3. Normalizar [0, 1] para uso no model.py
# ---------------------------------------------------------------------------

def normalise_profiles(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza cada perfil para [0, 1] dividindo pelo valor máximo horário.

    Nota: os valores ERSE são fracções anuais (soma = 1), não kWh.
    A normalização por max permite usar directamente como curve no model.py,
    que escala pelo peak_consumption_kw de cada agente.
    """
    df = df_hourly.copy()
    for col in ["BTN_A", "BTN_B", "BTN_C", "IP"]:
        max_val = df[col].max()
        df[f"{col}_norm"] = (df[col] / max_val).clip(0.0, 1.0)
        print(f"  {col}: max_horário={max_val:.6f} → normalizado [0, 1]")
    return df


# ---------------------------------------------------------------------------
# 4. Validação
# ---------------------------------------------------------------------------

def validate(df: pd.DataFrame) -> None:
    """Sanity checks sobre os perfis normalizados."""

    for col in ["BTN_A_norm", "BTN_C_norm"]:
        arr = df[col].to_numpy()

        assert len(arr) == 8760,     f"{col}: esperado 8760 horas"
        assert arr.min() >= 0.0,     f"{col}: valores negativos"
        assert arr.max() <= 1.0,     f"{col}: normalização falhou"
        assert arr.max() > 0.5,      f"{col}: pico suspeito ({arr.max():.4f})"

        # BTN A: pico deve ser no período noturno (18h-22h)
        # verificar que a média de pico > média de madrugada
        evening = df.loc[df["timestamp"].dt.hour.isin([18,19,20,21,22]), col].mean()
        night   = df.loc[df["timestamp"].dt.hour.isin([1,2,3,4]), col].mean()
        assert evening > night, f"{col}: padrão de pico inesperado"

    print("\n  ✓ Validação passou")
    for col in ["BTN_A_norm", "BTN_C_norm"]:
        arr = df[col].to_numpy()
        evening = df.loc[df["timestamp"].dt.hour.isin([18,19,20,21,22]), col].mean()
        night   = df.loc[df["timestamp"].dt.hour.isin([1,2,3,4]), col].mean()
        print(f"    {col}: pico noturno={evening:.4f} vs madrugada={night:.4f}")


# ---------------------------------------------------------------------------
# 5. Guardar outputs
# ---------------------------------------------------------------------------

def save_outputs(df: pd.DataFrame, source_path: Path) -> None:
    """
    Guarda três ficheiros em data/:
        consumption_curve_btn_a.npy  → array 1D para agentes 'residential'
        consumption_curve_btn_c.npy  → array 1D para agentes 'small_business'
        erse_hourly_2023.csv         → DataFrame completo (features para Fase 2)
        erse_metadata.json           → rastreabilidade
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Arrays numpy — substitutos directos de consumption_profile() no model.py
    for erse_col, agent_type in [("BTN_A_norm", "btn_a"), ("BTN_C_norm", "btn_c")]:
        arr      = df[erse_col].to_numpy(dtype=np.float32)
        npy_path = OUTPUT_DIR / f"consumption_curve_{agent_type}.npy"
        np.save(npy_path, arr)
        print(f"  ✓ Array numpy guardado: {npy_path}  shape={arr.shape}")

    # CSV horário completo — para feature engineering na Fase 2
    csv_cols = ["timestamp", "BTN_A", "BTN_B", "BTN_C", "BTN_A_norm", "BTN_C_norm"]
    csv_path = OUTPUT_DIR / "erse_hourly_2023.csv"
    df[csv_cols].to_csv(csv_path, index=False)
    print(f"  ✓ CSV horário guardado: {csv_path}")

    # Metadata
    meta = {
        "source"      : "E-REDES — Perfis de Consumo BTN 2023",
        "url"         : "https://www.e-redes.pt/pt-pt/perfis-de-consumo",
        "directive"   : "Diretiva ERSE nº 16/2023",
        "input_file"  : source_path.name,
        "n_hours"     : len(df),
        "profiles_used": list(PROFILE_MAP.keys()),
        "BTN_A_max_raw": round(float(df["BTN_A"].max()), 8),
        "BTN_C_max_raw": round(float(df["BTN_C"].max()), 8),
        "normalisation": "divided by hourly max → [0, 1]",
        "agent_mapping": PROFILE_MAP,
    }
    meta_path = OUTPUT_DIR / "erse_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Metadata guardada: {meta_path}")


# ---------------------------------------------------------------------------
# 6. Hint de integração com model.py
# ---------------------------------------------------------------------------

INTEGRATION_HINT = """
─────────────────────────────────────────────────────────
Como integrar no model.py (Fase 2):

No Household.__init__, substituir:
    self._consumption_curve = consumption_profile(8760, profile_type, agent_rng)

Por:
    _CURVES = {
        "residential"  : np.load("data/consumption_curve_btn_a.npy"),
        "small_business": np.load("data/consumption_curve_btn_c.npy"),
    }
    base_curve = _CURVES[profile_type]
    noise = agent_rng.uniform(0.93, 1.07, size=8760)  # variabilidade individual ±7%
    self._consumption_curve = np.clip(base_curve * noise, 0, 1)

Recomendação: carregar os arrays uma vez a nível de modelo (não por agente)
para evitar I/O repetido. Ver hint de integração completo no README.
─────────────────────────────────────────────────────────
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Processar perfis de consumo ERSE BTN para o modelo Mesa"
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help=f"Caminho para o ficheiro Excel da E-REDES (default: {DEFAULT_INPUT})"
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Ficheiro não encontrado: {args.input}\n"
            f"Descarrega o ficheiro em https://www.e-redes.pt/pt-pt/perfis-de-consumo\n"
            f"e coloca-o em {DEFAULT_INPUT}"
        )

    print("=" * 55)
    print("  ERSE BTN Profiles — CER Porto")
    print("=" * 55)

    # Pipeline
    print("\n[1/4] A carregar Excel...")
    df_qh = load_erse_excel(args.input)

    print("\n[2/4] A agregar para granularidade horária...")
    df_h = aggregate_to_hourly(df_qh)

    print("\n[3/4] A normalizar perfis...")
    df_norm = normalise_profiles(df_h)

    print("\n[4/4] A validar e guardar...")
    validate(df_norm)
    save_outputs(df_norm, args.input)

    print(INTEGRATION_HINT)


if __name__ == "__main__":
    main()
