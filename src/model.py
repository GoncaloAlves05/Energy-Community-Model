"""
Simulação de Comunidade de Energia Renovável (CER)
===================================================
Ferramenta: Mesa 3.x (Agent-Based Modelling em Python)
Referência legal: DL 15/2022 + RAC nº 815/2023 (ERSE)

Perguntas de investigação:
  P1. Como é que a composição da comunidade (% prosumers)
      afeta a taxa de autoconsumo coletivo e a poupança?
  P2. Como é que o modo de partilha de energia afeta
      os outcomes coletivos?

Granularidade temporal: 1 tick = 1 hora | 1 run = 8760 horas
"""

import numpy as np
import mesa

# ---------------------------------------------------------------------------
# Parâmetros económicos (€/kWh) — valores de referência PT 2024
# ---------------------------------------------------------------------------
GRID_PRICE_BUY  = 0.22   # preço de compra à rede (tarifa BTN)
GRID_PRICE_SELL = 0.05   # preço de venda de excedente (OMIE médio)
NETWORK_FEE_ACC = 0.01   # tarifa de acesso à rede para energia partilhada


# ---------------------------------------------------------------------------
# Perfis horários
# ---------------------------------------------------------------------------

def solar_profile(n_hours: int = 8760, rng=None) -> np.ndarray:
    """
    Perfil de produção solar horária normalizado entre 0 e 1.
    Baseado em dados PVGIS para Portugal (inclinação 30°).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    hours      = np.arange(n_hours)
    day_of_year = (hours // 24) % 365
    hour_of_day = hours % 24

    # Sazonalidade: pico no verão (dia 172 = 21 junho)
    seasonality = 0.5 + 0.5 * np.cos(2 * np.pi * (day_of_year - 172) / 365)

    # Curva em sino centrada às 13h (hora solar)
    solar_bell = np.exp(-0.5 * ((hour_of_day - 13) / 3.5) ** 2)
    solar_bell[(hour_of_day < 6) | (hour_of_day >= 20)] = 0.0

    profile = seasonality * solar_bell
    noise   = rng.uniform(0.85, 1.0, size=n_hours)
    profile = profile * noise

    return profile / (profile.max() + 1e-9)


def consumption_profile(n_hours: int = 8760, profile_type: str = "residential",
                        rng=None) -> np.ndarray:
    """
    Perfil de consumo horário normalizado entre 0 e 1.
    Tipos: 'residential' (BTN C) ou 'small_business' (BTN D).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    hour_of_day = np.arange(n_hours) % 24

    if profile_type == "residential":
        # Pico manhã (7-9h) e noite (18-22h)
        base = np.ones(n_hours) * 0.3
        base[np.isin(hour_of_day, [7, 8, 9])]              = 0.7
        base[np.isin(hour_of_day, [18, 19, 20, 21, 22])]   = 0.9
        base[np.isin(hour_of_day, [0, 1, 2, 3, 4, 5])]     = 0.15
    else:
        # Pico durante o dia (9h-18h)
        base = np.ones(n_hours) * 0.2
        base[np.isin(hour_of_day, list(range(9, 19)))]      = 0.85
        base[np.isin(hour_of_day, [0, 1, 2, 3, 4, 5])]     = 0.05

    noise   = rng.uniform(0.9, 1.1, size=n_hours)
    profile = base * noise

    return profile / profile.max()

# ---------------------------------------------------------------------------
# Agente: Household
# ---------------------------------------------------------------------------

class Household(mesa.Agent):
    """
    Membro da comunidade de energia.

    Parâmetros:
        is_prosumer      : tem painéis solares (UPAC)?
        peak_consumption : consumo máximo horário (kWh)
        peak_production  : produção máxima horária (kWh); 0 se consumidor puro
        profile_type     : 'residential' ou 'small_business'
    """

    def __init__(self, model, is_prosumer: bool, peak_consumption: float,
                 peak_production: float = 0.0, profile_type: str = "residential"):
        super().__init__(model)

        self.is_prosumer      = is_prosumer
        self.peak_consumption = peak_consumption
        self.peak_production  = peak_production
        self.profile_type     = profile_type

        # Gerar perfis anuais com variabilidade individual
        agent_rng = np.random.default_rng(self.unique_id)
        self._consumption_curve = consumption_profile(8760, profile_type, agent_rng)
        self._solar_curve       = solar_profile(8760, agent_rng) if is_prosumer else np.zeros(8760)

        # Estado corrente (atualizado a cada tick)
        self.consumption       = 0.0
        self.production        = 0.0
        self.shared_received   = 0.0
        self.grid_import       = 0.0
        self.savings_tick      = 0.0
        self.cumulative_savings = 0.0

    def step(self):
        """Calcular produção e consumo para o tick atual."""
        t = self.model.current_tick % 8760
        self.consumption = self.peak_consumption * self._consumption_curve[t]
        self.production  = self.peak_production  * self._solar_curve[t] if self.is_prosumer else 0.0

        # Partilha e poupança são calculadas pelo EGAC depois
        self.shared_received = 0.0
        self.grid_import     = 0.0
        self.savings_tick    = 0.0

    def settle(self, shared_kWh: float):
        """
        Chamado pelo EGAC após calcular a partilha.
        Calcula importação residual da rede e poupança económica.
        """
        self.shared_received = shared_kWh
        net_need             = self.consumption - self.production - self.shared_received
        self.grid_import     = max(0.0, net_need)

        # Poupança = custo sem comunidade - custo com comunidade
        cost_without = self.consumption * GRID_PRICE_BUY
        cost_with    = (self.grid_import * GRID_PRICE_BUY
                        + self.shared_received * (GRID_PRICE_BUY - NETWORK_FEE_ACC))

        self.savings_tick       = cost_without - cost_with
        self.cumulative_savings += self.savings_tick

# ---------------------------------------------------------------------------
# Modos de partilha (RAC — Regulamento de Acesso às Redes, nº 815/2023, ERSE)
# ---------------------------------------------------------------------------

SHARING_MODES = ["fixed", "proportional", "hierarchical", "dynamic"]


def share_energy(total_production: float, households: list, mode: str) -> dict:
    """
    Distribui a produção coletiva pelos membros segundo o modo de partilha.
    Retorna dict {unique_id: kWh_recebido}.
    """
    allocation = {h.unique_id: 0.0 for h in households}
    available  = total_production

    if available <= 0:
        return allocation

    if mode == "fixed":
        # Coeficientes iguais para todos os membros
        share_per_member = available / len(households)
        for h in households:
            deficit = max(0.0, h.consumption - h.production)
            allocation[h.unique_id] = min(share_per_member, deficit)

    elif mode == "proportional":
        # Proporcional ao consumo medido no período
        total_consumption = sum(h.consumption for h in households)
        if total_consumption <= 0:
            return allocation
        for h in households:
            coef    = h.consumption / total_consumption
            deficit = max(0.0, h.consumption - h.production)
            allocation[h.unique_id] = min(coef * available, deficit)

    elif mode == "hierarchical":
        # Prioridade por ordem de adesão (unique_id crescente)
        sorted_h  = sorted(households, key=lambda h: h.unique_id)
        remaining = available
        for h in sorted_h:
            if remaining <= 0:
                break
            deficit = max(0.0, h.consumption - h.production)
            given   = min(deficit, remaining)
            allocation[h.unique_id] = given
            remaining -= given

    elif mode == "dynamic":
        # Proporcional ao défice individual (consumo - produção própria)
        deficits      = {h.unique_id: max(0.0, h.consumption - h.production)
                         for h in households}
        total_deficit = sum(deficits.values())
        if total_deficit <= 0:
            return allocation
        for h in households:
            coef = deficits[h.unique_id] / total_deficit
            allocation[h.unique_id] = min(coef * available, deficits[h.unique_id])

    return allocation


# ---------------------------------------------------------------------------
# Agente: EGAC (Entidade Gestora do Autoconsumo Coletivo)
# ---------------------------------------------------------------------------

class EGAC(mesa.Agent):
    """
    Entidade Gestora do Autoconsumo Coletivo.
    Orquestra a partilha de energia em cada tick e acumula métricas.
    """

    def __init__(self, model, sharing_mode: str = "proportional"):
        super().__init__(model)
        self.sharing_mode = sharing_mode

        # Métricas acumuladas ao longo do run
        self.total_production  = 0.0
        self.total_consumption = 0.0
        self.total_shared      = 0.0
        self.total_grid_export = 0.0
        self.total_grid_import = 0.0

    @property
    def self_consumption_rate(self) -> float:
        """Energia partilhada / energia produzida."""
        if self.total_production <= 0:
            return 0.0
        return self.total_shared / self.total_production

    @property
    def self_sufficiency_rate(self) -> float:
        """Energia partilhada / energia consumida."""
        if self.total_consumption <= 0:
            return 0.0
        return self.total_shared / self.total_consumption

    def step(self):
        households = list(self.model.agents_by_type[Household])

        # Totais do tick
        prod_total = sum(h.production for h in households)
        cons_total = sum(h.consumption for h in households)

        # Distribuir energia
        allocation = share_energy(prod_total, households, self.sharing_mode)

        # Liquidar cada household
        for h in households:
            h.settle(allocation[h.unique_id])

        # Acumular métricas
        shared_total = sum(allocation.values())
        self.total_production  += prod_total
        self.total_consumption += cons_total
        self.total_shared      += shared_total
        self.total_grid_export += max(0.0, prod_total - shared_total)
        self.total_grid_import += sum(h.grid_import for h in households)

# ---------------------------------------------------------------------------
# Modelo: EnergyCommunity
# ---------------------------------------------------------------------------

class EnergyCommunity(mesa.Model):
    """
    Comunidade de Energia Renovável (CER) — DL 15/2022 (Portugal).

    Parâmetros:
        n_members           : número total de membros
        prosumer_ratio      : fração de membros com UPAC [0, 1]
        sharing_mode        : modo de partilha (ver SHARING_MODES)
        n_hours             : duração da simulação em horas (default: 8760 = 1 ano)
        peak_consumption_kw : consumo de ponta médio por membro (kW)
        peak_production_kw  : produção de ponta média por prosumer (kW)
        rng                 : semente aleatória (para reprodutibilidade)
    """

    def __init__(self,
                 n_members: int = 20,
                 prosumer_ratio: float = 0.5,
                 sharing_mode: str = "proportional",
                 n_hours: int = 8760,
                 peak_consumption_kw: float = 1.5,
                 peak_production_kw: float = 3.0,
                 rng: int = 42):
        super().__init__(rng=rng)

        self.n_members      = n_members
        self.prosumer_ratio = prosumer_ratio
        self.sharing_mode   = sharing_mode
        self.n_hours        = n_hours
        self.current_tick   = 0

        # Criar membros
        n_prosumers = round(n_members * prosumer_ratio)
        for i in range(n_members):
            is_prosumer = i < n_prosumers
            indiv_cons  = peak_consumption_kw * self.random.uniform(0.8, 1.2)
            indiv_prod  = peak_production_kw  * self.random.uniform(0.8, 1.2) if is_prosumer else 0.0
            profile     = "small_business" if (i % 5 == 0) else "residential"
            Household(self,
                      is_prosumer=is_prosumer,
                      peak_consumption=indiv_cons,
                      peak_production=indiv_prod,
                      profile_type=profile)

        # Criar EGAC
        self.egac = EGAC(self, sharing_mode=sharing_mode)

        # DataCollector — recolhe métricas acumuladas a cada tick
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "self_consumption_rate" : lambda m: m.egac.self_consumption_rate,
                "self_sufficiency_rate" : lambda m: m.egac.self_sufficiency_rate,
                "total_production_kWh"  : lambda m: m.egac.total_production,
                "total_shared_kWh"      : lambda m: m.egac.total_shared,
                "total_grid_export_kWh" : lambda m: m.egac.total_grid_export,
                "total_grid_import_kWh" : lambda m: m.egac.total_grid_import,
                "avg_savings_eur"       : lambda m: np.mean(
                    [h.cumulative_savings for h in m.agents_by_type[Household]]
                ),
            }
        )

    def step(self):
        # 1. Households calculam produção e consumo
        self.agents_by_type[Household].do("step")
        # 2. EGAC distribui energia e liquida poupanças
        self.egac.step()
        # 3. Recolher métricas
        self.datacollector.collect(self)
        self.current_tick += 1

    def run(self) -> dict:
        """Corre a simulação completa e devolve métricas finais."""
        for _ in range(self.n_hours):
            self.step()
        return self._summary()

    def _summary(self) -> dict:
        """Métricas finais do run."""
        households = list(self.agents_by_type[Household])
        savings    = [h.cumulative_savings for h in households]
        return {
            "n_members"             : self.n_members,
            "prosumer_ratio"        : self.prosumer_ratio,
            "sharing_mode"          : self.sharing_mode,
            "self_consumption_rate" : round(self.egac.self_consumption_rate, 4),
            "self_sufficiency_rate" : round(self.egac.self_sufficiency_rate, 4),
            "total_production_kWh"  : round(self.egac.total_production, 1),
            "total_shared_kWh"      : round(self.egac.total_shared, 1),
            "total_grid_export_kWh" : round(self.egac.total_grid_export, 1),
            "total_grid_import_kWh" : round(self.egac.total_grid_import, 1),
            "avg_savings_eur"       : round(float(np.mean(savings)), 2),
            "min_savings_eur"       : round(float(np.min(savings)), 2),
            "max_savings_eur"       : round(float(np.max(savings)), 2),
        }