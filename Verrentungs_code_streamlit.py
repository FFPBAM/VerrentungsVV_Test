"""
Verrentungs_code_streamlit.py

Flexible engine for:
    reading MSCI World + REXP + CPI from an Excel file
    building nominal and real return series with optional fees
    constructing a portfolio return series from configurable weights
    running a constant-withdrawal simulation with configurable rate and horizon
    modelling capital gains tax on realised gains via a cost-basis approach
    producing charts (fan chart, success curve, distribution)
    exporting all historical cohort paths and a cohort summary table

All important scenario parameters are configured once in the SCENARIO SETTINGS
section below.

This file contains both:
    1) The full engine (functions + plotting helpers)
    2) A Streamlit front-end (no separate engine file required)

VARIANTE B (IMPORTIERBAR FÜR TESTS):
    Der komplette Streamlit-Code ist in run_app() gekapselt.
    Beim Import (z.B. durch pytest) wird die UI NICHT ausgeführt.

Run with:
    streamlit run Verrentungs_code_streamlit.py
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from matplotlib.ticker import FuncFormatter
from matplotlib.figure import Figure

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# SCENARIO SETTINGS (ADJUST HERE)
# ---------------------------------------------------------------------------

# Path to Excel file with MSCI World, REXP, CPI
DATA_FILE = Path(r"Daten Verrentung.xlsx")

# Portfolio weights (must sum to 1.0)
PORTFOLIO_WEIGHTS = {
    "msci_world": 0.60,
    "rexp": 0.35,
    "Gold": 0.05,
}

# Product fee
ANNUAL_FEE = 0.0184      # 1.8 % p.a.
APPLY_FEES = True       # True: use net-of-fee returns; False: ignore fees

# Investor capital gains tax (German Abgeltungsteuer, simplified)
CAPITAL_GAINS_TAX_RATE = 0.25   # 25 % auf realisierte Kursgewinne
APPLY_TAX = True               # True: tax realised gains on sales

# Work in real (inflation-adjusted) or nominal terms
USE_INFLATION = True           # True: real, False: nominal

# Withdrawal plan
WITHDRAWAL_RATE = 0.05          # 4 % per year net to the investor
INITIAL_WEALTH = 1_000_000.0    # starting capital (can be scaled)
HORIZON_YEARS = 30
PERIODS_PER_YEAR = 12           # monthly data

# Success-rate curve settings
SUCCESS_RATE_MIN = 0.02         # min withdrawal rate for curve (2 %)
SUCCESS_RATE_MAX = 0.08         # max withdrawal rate for curve (10 %)
SUCCESS_RATE_STEP = 0.0025      # step (0.25 %-points)


# ---------------------------------------------------------------------------
# CONFIGURATION DATACLASSES
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """
    Configuration for the input data.

    asset_columns maps internal asset names to column names in the Excel sheet.
    """
    excel_path: Path
    sheet_name: str = "Import_Daten"
    date_column: str = "Dates"
    cpi_column: str = "Inflation DE"
    asset_columns: Dict[str, str] = field(
        default_factory=lambda: {
            "msci_world": "NDDUWI Index",
            "rexp": "REXP Index",
            "Gold": "Gold",
        }
    )


@dataclass
class PortfolioConfig:
    """
    Configuration for the portfolio construction.

    name          : label for the resulting portfolio return series.
    weights       : asset weights, sum should be 1.0.
    annual_fee    : annual fee in decimal (0.018 for 1.8 % p.a.).
    use_fees      : if True, use net-of-fee returns; else gross returns.
    use_inflation : if True, use real returns (deflated by CPI);
                    if False, use nominal returns.
    """
    name: str
    weights: Dict[str, float]
    annual_fee: float = 0.0
    use_fees: bool = True
    use_inflation: bool = True


@dataclass
class SimulationConfig:
    """
    Configuration for the withdrawal simulation.

    annual_withdrawal_rate : constant annual net withdrawal rate (0.04 = 4 %).
    initial_wealth         : starting capital.
    periods_per_year       : 12 for monthly data.
    horizon_years          : length of the retirement in years; if None,
                             the full history is used.
    """
    annual_withdrawal_rate: float = 0.04
    initial_wealth: float = 100.0
    periods_per_year: int = 12
    horizon_years: Optional[int] = 30


# ---------------------------------------------------------------------------
# DATA LOADING AND NOMINAL RETURNS
# ---------------------------------------------------------------------------

def load_market_data(cfg: DataConfig) -> pd.DataFrame:
    """
    Load MSCI World, REXP, Gold and CPI from the Excel file and align them.

    Returns a DataFrame with columns:
        msci_world, rexp, gold, cpi
    and a monthly DateTimeIndex.
    """
    df = pd.read_excel(cfg.excel_path, sheet_name=cfg.sheet_name)

    df[cfg.date_column] = pd.to_datetime(df[cfg.date_column])
    df = df.set_index(cfg.date_column).sort_index()

    cols = {}

    for asset_name, col_name in cfg.asset_columns.items():
        if col_name not in df.columns:
            raise KeyError(f"Column {col_name!r} for asset {asset_name!r} not found")
        cols[asset_name] = df[col_name]

    if cfg.cpi_column not in df.columns:
        raise KeyError(f"CPI column {cfg.cpi_column!r} not found")
    cols["cpi"] = df[cfg.cpi_column]

    panel = pd.DataFrame(cols).dropna()

    return panel


def compute_nominal_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly nominal returns for all assets plus monthly inflation.

    For each asset column:
        r_t = level_t / level_{t-1} - 1

    Inflation is computed as percent change in CPI.
    """
    asset_names = [c for c in panel.columns if c != "cpi"]

    asset_levels = panel[asset_names]
    rets = asset_levels.pct_change().dropna()

    cpi = panel["cpi"].pct_change().dropna()
    cpi = cpi.reindex(rets.index)

    result = rets.copy()
    result["inflation"] = cpi

    return result


# ---------------------------------------------------------------------------
# FEES AND REAL RETURNS
# ---------------------------------------------------------------------------

def apply_annual_fee_to_returns(
    rets: pd.DataFrame,
    annual_fee: float,
    asset_cols=None,
) -> pd.DataFrame:
    """
    Apply a constant annual fee to monthly returns for the given assets.

    Fee model:
        net_return_t = (1 + gross_return_t) * (1 - monthly_fee) - 1

    The annual fee is converted into an equivalent constant monthly fee.
    """
    if asset_cols is None:
        asset_cols = [c for c in rets.columns if c != "inflation"]

    monthly_fee = 1.0 - (1.0 - annual_fee) ** (1.0 / 12.0)

    out = rets.copy()

    for col in asset_cols:
        if col == "inflation":
            continue
        gross = 1.0 + out[col]
        out[col + "_net"] = gross * (1.0 - monthly_fee) - 1.0

    return out


def make_real_returns(rets: pd.DataFrame, use_net: bool = False) -> pd.DataFrame:
    """
    Convert nominal returns to real returns using the 'inflation' column.

    If use_net is False:
        works on gross columns and creates *_real columns.

    If use_net is True:
        works on *_net columns and creates *_real_net columns.

    Real return formula:
        1 + r_real = (1 + r_nominal) / (1 + inflation)
    """
    infl = rets["inflation"]

    base_cols = []
    for col in rets.columns:
        if col == "inflation":
            continue
        is_net = col.endswith("_net")
        if use_net and is_net:
            base_cols.append(col)
        elif (not use_net) and (not is_net):
            base_cols.append(col)

    real = {}

    for col in base_cols:
        r = rets[col]
        base_name = col.replace("_net", "")
        suffix = "_real_net" if col.endswith("_net") else "_real"
        real_name = base_name + suffix
        real[real_name] = (1.0 + r) / (1.0 + infl) - 1.0

    return pd.DataFrame(real, index=rets.index)


# ---------------------------------------------------------------------------
# PORTFOLIO CONSTRUCTION
# ---------------------------------------------------------------------------

def build_portfolio_returns(
    rets: pd.DataFrame,
    weights: Dict[str, float],
    use_net: bool,
    use_real: bool,
    portfolio_name: str,
) -> pd.Series:
    """
    Build a portfolio return series from asset return columns.

    Depending on use_net and use_real it selects:
        asset_real_net, asset_real, asset_net or asset.
    """
    columns_to_use: Dict[str, float] = {}

    for asset, w in weights.items():
        if use_real and use_net:
            col = f"{asset}_real_net"
        elif use_real and not use_net:
            col = f"{asset}_real"
        elif (not use_real) and use_net:
            col = f"{asset}_net"
        else:
            col = asset

        if col not in rets.columns:
            raise KeyError(f"Expected column {col!r} for asset {asset!r} not found")
        columns_to_use[col] = w

    port = pd.Series(0.0, index=rets.index, name=portfolio_name)

    for col, w in columns_to_use.items():
        port = port + w * rets[col]

    return port


def prepare_portfolio_returns(
    panel: pd.DataFrame,
    portfolio_cfg: PortfolioConfig,
) -> pd.Series:
    """
    From raw index levels to a ready-to-use portfolio return series.

    Steps:
        1) Compute nominal returns and inflation.
        2) Optionally apply annual fee (adds *_net columns).
        3) Compute real returns (adds *_real and *_real_net).
        4) Build portfolio returns according to PortfolioConfig.

    STREAMLIT-APP-ZUSATZ:
        In der Berater-App erlauben wir Gewichte < 100 % (Rest = Liquidität).
        Damit der Rest fachlich konsistent behandelt wird, fügen wir hier eine
        künstliche Renditereihe "cash" (= Liquidität) hinzu.

        Nominal: 0 % Rendite
        Real: wird automatisch zu -Inflation, da (1+0)/(1+Inflation)-1 = -Inflation
    """
    rets = compute_nominal_returns(panel)

    # Liquidität / Cash-Rendite ergänzen, damit Gewichte unter 100 % sauber funktionieren
    if "cash" not in rets.columns:
        rets["cash"] = 0.0

    if portfolio_cfg.use_fees and portfolio_cfg.annual_fee > 0:
        rets = apply_annual_fee_to_returns(rets, portfolio_cfg.annual_fee)

    real_gross = make_real_returns(rets, use_net=False)
    rets_full = rets.join(real_gross)

    if portfolio_cfg.use_fees and portfolio_cfg.annual_fee > 0:
        real_net = make_real_returns(rets, use_net=True)
        rets_full = rets_full.join(real_net)

    port_rets = build_portfolio_returns(
        rets_full,
        weights=portfolio_cfg.weights,
        use_net=portfolio_cfg.use_fees,
        use_real=portfolio_cfg.use_inflation,
        portfolio_name=portfolio_cfg.name,
    )

    return port_rets


# ---------------------------------------------------------------------------
# WITHDRAWAL SIMULATION WITH COST-BASIS TAX
# ---------------------------------------------------------------------------

def simulate_constant_withdrawal(
    portfolio_returns: pd.Series,
    sim_cfg: SimulationConfig,
    tax_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Simulate a constant-withdrawal strategy with optional capital gains tax.

    Interpretation:

    portfolio_returns
        pre-tax portfolio returns (already net of product fees and optionally
        inflation-adjusted).

    tax_rate
        flat tax rate on realised capital gains (0.25 = 25 %).
        If 0.0, no investor-level tax is applied.

    sim_cfg
        defines annual withdrawal rate, horizon, initial wealth and frequency.

    Tax model (stylised German Abgeltungsteuer):

        1) Portfolio grows or falls with the pre-tax return each period.
        2) Then we sell units so that the investor receives a fixed NET
           withdrawal amount (withdraw_per_period) after tax.
        3) The sale contains principal and gain. We assume gains are spread
           uniformly over the portfolio (proportional sale).
        4) Realised gain is taxed at 'tax_rate'. Tax is paid from the sale
           proceeds, so the portfolio value reduces by the gross sale.
        5) We track the tax basis (= remaining principal) alongside wealth.

    Output columns:
        wealth          : portfolio value after tax and withdrawal
        basis           : remaining tax basis (principal)
        withdrawal_net  : net cash to investor in this period
        tax_paid        : tax paid in this period
        gross_sale      : gross sale value
        return          : pre-tax portfolio return for this period
    """
    if sim_cfg.horizon_years is not None:
        max_periods = sim_cfg.horizon_years * sim_cfg.periods_per_year
        returns = portfolio_returns.iloc[:max_periods]
    else:
        returns = portfolio_returns

    withdraw_per_period = (
        sim_cfg.initial_wealth
        * sim_cfg.annual_withdrawal_rate
        / sim_cfg.periods_per_year
    )

    wealth_values = []
    basis_values = []
    withdrawals_net = []
    taxes_paid = []
    gross_sales = []

    wealth = sim_cfg.initial_wealth
    basis = sim_cfg.initial_wealth

    for r in returns:
        if wealth <= 0:
            wealth_values.append(0.0)
            basis_values.append(0.0)
            withdrawals_net.append(0.0)
            taxes_paid.append(0.0)
            gross_sales.append(0.0)
            wealth = 0.0
            basis = 0.0
            continue

        # Portfolio evolves with pre-tax return
        wealth_pre = wealth * (1.0 + r)

        if tax_rate <= 0.0:
            # Simple no-tax case: we just withdraw the target amount
            sale = min(withdraw_per_period, wealth_pre)
            tax = 0.0
            net_withdraw = sale

            principal_fraction = (basis / wealth_pre) if wealth_pre > 0 else 0.0
            principal_sold = sale * principal_fraction
            wealth = wealth_pre - sale
            basis = max(basis - principal_sold, 0.0)

        else:
            # Split wealth into principal and gains
            if wealth_pre > basis:
                total_gain = wealth_pre - basis
                gain_fraction = total_gain / wealth_pre
            else:
                total_gain = 0.0
                gain_fraction = 0.0

            principal_fraction = 1.0 - gain_fraction

            # We want net withdrawal = withdraw_per_period
            # sale * (1 - tax_rate * gain_fraction) = withdraw_per_period
            if gain_fraction > 0:
                net_factor = 1.0 - tax_rate * gain_fraction
            else:
                net_factor = 1.0

            if net_factor <= 0.0:
                sale = wealth_pre
            else:
                sale = withdraw_per_period / net_factor

            sale = min(sale, wealth_pre)

            realised_gain = sale * gain_fraction
            tax = tax_rate * realised_gain
            net_withdraw = sale - tax

            principal_sold = sale * principal_fraction
            wealth = wealth_pre - sale
            basis = max(basis - principal_sold, 0.0)

        wealth_values.append(wealth)
        basis_values.append(basis)
        withdrawals_net.append(net_withdraw)
        taxes_paid.append(tax)
        gross_sales.append(sale)

    path = pd.DataFrame(
        {
            "wealth": wealth_values,
            "basis": basis_values,
            "withdrawal_net": withdrawals_net,
            "tax_paid": taxes_paid,
            "gross_sale": gross_sales,
            "return": returns.values,
        },
        index=returns.index,
    )

    return path


# ---------------------------------------------------------------------------
# HELPERS FOR TITLES AND LABELS
# ---------------------------------------------------------------------------

def pretty_asset_name(asset: str) -> str:
    mapping = {
        "msci_world": "MSCI Welt",
        "rexp": "REXP",
        "Gold": "Gold",
        "cash": "Liquidität",
    }
    return mapping.get(asset, asset)


def format_weights(weights: Dict[str, float]) -> str:
    parts = []
    for asset, w in weights.items():
        if w <= 1e-6:
            continue
        parts.append(f"{w*100:.0f} % {pretty_asset_name(asset)}")
    return ", ".join(parts)


def format_chart_title(
    prefix: str,
    port_cfg: PortfolioConfig,
    sim_cfg: SimulationConfig,
    extra: str = "",
) -> str:
    """
    Build a chart title that reflects the current configuration.
    """
    w_str = format_weights(port_cfg.weights)

    if port_cfg.use_fees and port_cfg.annual_fee > 0:
        fee_str = f"{port_cfg.annual_fee*100:.2f} % Gebühren"
    else:
        fee_str = "ohne Gebühren"

    infl_str = "inflationsbereinigt" if port_cfg.use_inflation else "nominal"

    base = (
        f"{prefix}: {w_str}, "
        f"{sim_cfg.annual_withdrawal_rate*100:.1f} % Entnahme p.a., "
        f"{fee_str}, {infl_str}"
    )

    if extra:
        base += f", {extra}"

    return base


# ---------------------------------------------------------------------------
# STYLE SETTINGS (company colours, fonts, figure size)
# ---------------------------------------------------------------------------
#
# GRAFIK-KONFIGURATION (ZENTRALER ORT FÜR DESIGN-ANPASSUNGEN)
#
# Wenn ihr die Charts an euer Corporate Design anpassen wollt, dann ist dieser
# Block die wichtigste Stelle im Code.
#
# Typische Anpassungen in der Praxis:
#   1) Farben: COMPANY_BLUE, BAND_50_COLOR, BAND_80_COLOR, BAND_100_COLOR
#   2) Linienfarben: MEDIAN_COLOR, WORST_COLOR, BEST_COLOR
#   3) Schrift: FONT_FAMILY und die FONT_SIZE_* Werte
#   4) Abmessungen: FIGSIZE_16_9 (z.B. für Beratung auf 16:9 Bildschirmen)
#   5) Auflösung: DPI_EXPORT
#
# Alle Plot-Funktionen verwenden diese globalen Einstellungen automatisch.
# ---------------------------------------------------------------------------

COMPANY_BLUE = "#003c71"
BAND_50_COLOR = "#7da6d8"
BAND_80_COLOR = "#c0d2ea"
BAND_100_COLOR = "#edf1f7"

MEDIAN_COLOR = COMPANY_BLUE
WORST_COLOR = "#b85c5c"
BEST_COLOR = "#2e7d32"

FIGSIZE_16_9 = (10, 5.625)
DPI_EXPORT = 200

FONT_FAMILY = "DejaVu Sans"
FONT_SIZE_BASE = 11
FONT_SIZE_TITLE = 8
FONT_SIZE_LABEL = 12
FONT_SIZE_TICKS = 10
FONT_SIZE_LEGEND = 9
FONT_SIZE_ANNOT = 9

plt.rcParams.update(
    {
        "font.family": FONT_FAMILY,
        "font.size": FONT_SIZE_BASE,
        "axes.titlesize": FONT_SIZE_TITLE,
        "axes.labelsize": FONT_SIZE_LABEL,
        "xtick.labelsize": FONT_SIZE_TICKS,
        "ytick.labelsize": FONT_SIZE_TICKS,
        "legend.fontsize": FONT_SIZE_LEGEND,
    }
)


def euro_formatter(x, pos):
    """
    Format axis ticks as Euro amounts: 1.000.000 €
    """
    return f"{x:,.0f} €".replace(",", ".")


def set_euro_yaxis(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(euro_formatter))


# ---------------------------------------------------------------------------
# CHART FUNCTIONS
# ---------------------------------------------------------------------------

def build_wealth_matrix(
    portfolio_returns: pd.Series,
    sim_cfg: SimulationConfig,
    tax_rate: float,
) -> pd.DataFrame:
    """
    Build a wealth matrix for all rolling cohorts.

    Each column = one cohort (identified by its start date).
    Each row   = one month in retirement (1 .. horizon * periods_per_year).

    The number of cohorts is stored in matrix.attrs["n_cohorts"].
    """
    periods = sim_cfg.horizon_years * sim_cfg.periods_per_year
    wealth_paths = []
    start_dates = []

    n_cohorts = len(portfolio_returns) - periods + 1

    for start in range(n_cohorts):
        window = portfolio_returns.iloc[start:start + periods]
        start_dates.append(window.index[0])

        cfg = SimulationConfig(
            annual_withdrawal_rate=sim_cfg.annual_withdrawal_rate,
            initial_wealth=sim_cfg.initial_wealth,
            periods_per_year=sim_cfg.periods_per_year,
            horizon_years=sim_cfg.horizon_years,
        )
        path = simulate_constant_withdrawal(window, cfg, tax_rate=tax_rate)
        wealth_paths.append(path["wealth"].values)

    matrix = pd.DataFrame(wealth_paths).T
    matrix.index = range(1, periods + 1)
    matrix.index.name = "month_in_retirement"
    matrix.columns = pd.to_datetime(start_dates)
    matrix.columns.name = "start_date"
    matrix.attrs["n_cohorts"] = n_cohorts

    return matrix


def summarise_cohorts(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Create a table with one row per cohort (start date).
    """
    start_dates = matrix.columns

    terminal_wealth = matrix.iloc[-1]
    min_wealth = matrix.min(axis=0)
    max_wealth = matrix.max(axis=0)
    ruined_mask = (matrix == 0.0).any(axis=0)

    ruin_month = []
    for col in matrix.columns:
        series = matrix[col]
        zero_idx = (series <= 0.0).to_numpy().nonzero()[0]
        ruin_month.append(int(zero_idx[0] + 1) if len(zero_idx) > 0 else None)

    ruin_year = [
        (m / 12.0) if m is not None else None
        for m in ruin_month
    ]

    summary = pd.DataFrame(
        {
            "start_date": start_dates,
            "terminal_wealth": terminal_wealth.values,
            "min_wealth": min_wealth.values,
            "max_wealth": max_wealth.values,
            "ruined": ruined_mask.values,
            "ruin_month": ruin_month,
            "ruin_year": ruin_year,
        }
    )

    summary = summary.sort_values("terminal_wealth", ascending=False).reset_index(drop=True)
    summary.attrs["n_cohorts"] = matrix.attrs.get("n_cohorts", len(summary))

    return summary


def plot_fan_chart(matrix: pd.DataFrame, title: str) -> Figure:
    """
    Plot percentile bands and extreme paths of wealth over the retirement horizon,
    styled with company colours and legend including min/max/mean/median
    (bezogen auf das Endvermögen der Kohorten).
    """
    p0 = matrix.min(axis=1)
    p10 = matrix.quantile(0.10, axis=1)
    p25 = matrix.quantile(0.25, axis=1)
    p50 = matrix.quantile(0.50, axis=1)
    p75 = matrix.quantile(0.75, axis=1)
    p90 = matrix.quantile(0.90, axis=1)
    p100 = matrix.max(axis=1)

    months = matrix.index.values
    years = months / 12.0

    terminals = matrix.iloc[-1]
    tw_min = terminals.min()
    tw_max = terminals.max()
    tw_mean = terminals.mean()
    tw_median = terminals.median()

    start_min = terminals.idxmin()
    start_max = terminals.idxmax()
    median_start = (terminals - tw_median).abs().idxmin()

    def fmt_ym(d):
        return d.strftime("%Y-%m") if isinstance(d, pd.Timestamp) else str(d)

    label_min = f"Schlechtester Verlauf: {euro_formatter(tw_min, None)} ({fmt_ym(start_min)})"
    label_max = f"Bester Verlauf: {euro_formatter(tw_max, None)} ({fmt_ym(start_max)})"
    label_median = f"Median: {euro_formatter(tw_median, None)} ({fmt_ym(median_start)})"
    label_mean = f"Mittelwert: {euro_formatter(tw_mean, None)}"

    fig, ax = plt.subplots(figsize=FIGSIZE_16_9, dpi=DPI_EXPORT)

    ax.fill_between(years, p0, p100, color=BAND_100_COLOR, alpha=1.0, label="100 % Band (min–max)")
    ax.fill_between(years, p10, p90, color=BAND_80_COLOR, alpha=1.0, label="80 % Band")
    ax.fill_between(years, p25, p75, color=BAND_50_COLOR, alpha=1.0, label="50 % Band")

    ax.plot(years, p50, color=MEDIAN_COLOR, linewidth=2.0, label=label_median)
    ax.plot(years, p0, color=WORST_COLOR, linestyle="--", linewidth=1.3, label=label_min)
    ax.plot(years, p100, color=BEST_COLOR, linestyle="--", linewidth=1.3, label=label_max)

    ax.plot([], [], color="grey", linestyle=":", linewidth=1.3, label=label_mean)

    n_cohorts = matrix.attrs.get("n_cohorts")
    if n_cohorts is not None:
        ax.text(
            0.99,
            0.02,
            f"{n_cohorts} historische Läufe",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=FONT_SIZE_ANNOT,
        )

    ax.set_title(title)
    ax.set_xlabel("Jahre im Ruhestand")
    ax.set_ylabel("Vermögen")
    set_euro_yaxis(ax)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_all_wealth_paths(matrix: pd.DataFrame, title: str) -> Figure:
    """
    Plot the wealth time series for all historical cohorts.

    Each column in 'matrix' is one cohort (one start date).
    Each row is one month in retirement.

    This produces a "Spaghetti-Chart" of all historical paths.
    The median path is highlighted for orientation.
    """
    months = matrix.index.values
    years = months / 12.0
    n_cohorts = matrix.attrs.get("n_cohorts", matrix.shape[1])

    fig, ax = plt.subplots(figsize=FIGSIZE_16_9, dpi=DPI_EXPORT)

    for col in matrix.columns:
        ax.plot(years, matrix[col].values, alpha=0.12, linewidth=1, color=BAND_50_COLOR)

    median_path = matrix.median(axis=1)
    ax.plot(years, median_path.values, linewidth=2.0, color=COMPANY_BLUE, label="Median")

    ax.set_title(title)
    ax.set_xlabel("Jahre im Ruhestand")
    ax.set_ylabel("Vermögen")
    set_euro_yaxis(ax)
    ax.legend()

    ax.text(
        0.99,
        0.02,
        f"{n_cohorts} historische Läufe",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=FONT_SIZE_ANNOT,
    )

    plt.tight_layout()
    return fig


def success_rate_for_withdrawal(
    portfolio_returns: pd.Series,
    annual_withdrawal_rate: float,
    sim_cfg_template: SimulationConfig,
    tax_rate: float,
) -> float:
    """
    Compute the fraction of cohorts that survive the full horizon
    for a given withdrawal rate and tax setting.
    """
    periods = sim_cfg_template.horizon_years * sim_cfg_template.periods_per_year
    n_cohorts = len(portfolio_returns) - periods + 1
    successes = 0

    for start in range(n_cohorts):
        window = portfolio_returns.iloc[start:start + periods]
        cfg = SimulationConfig(
            annual_withdrawal_rate=annual_withdrawal_rate,
            initial_wealth=sim_cfg_template.initial_wealth,
            periods_per_year=sim_cfg_template.periods_per_year,
            horizon_years=sim_cfg_template.horizon_years,
        )
        path = simulate_constant_withdrawal(window, cfg, tax_rate=tax_rate)
        if (path["wealth"] > 0).all():
            successes += 1

    return successes / n_cohorts if n_cohorts > 0 else float("nan")


def plot_success_curve(
    gross_returns: pd.Series,
    net_returns: pd.Series,
    sim_cfg: SimulationConfig,
    port_cfg_net: PortfolioConfig,
    rate_min: float,
    rate_max: float,
    rate_step: float,
    tax_rate: float,
    show_gross_line: bool = False,
) -> Figure:
    """
    Plot success probability versus withdrawal rate.

    net_returns: portfolio mit Gebühren
    gross_returns: portfolio ohne Gebühren (nur genutzt, wenn show_gross_line=True)
    """
    rates = np.arange(rate_min, rate_max + 1e-9, rate_step)

    success_net = [
        success_rate_for_withdrawal(net_returns, r, sim_cfg, tax_rate=tax_rate)
        for r in rates
    ]

    success_gross = None
    if show_gross_line:
        success_gross = [
            success_rate_for_withdrawal(gross_returns, r, sim_cfg, tax_rate=tax_rate)
            for r in rates
        ]

    periods = sim_cfg.horizon_years * sim_cfg.periods_per_year
    n_cohorts = len(net_returns) - periods + 1

    x_vals = rates * 100.0
    success_net_pct = np.array(success_net) * 100.0
    success_gross_pct = (
        np.array(success_gross) * 100.0
        if show_gross_line and success_gross is not None
        else None
    )

    fig, ax = plt.subplots(figsize=FIGSIZE_16_9, dpi=DPI_EXPORT)

    if show_gross_line and success_gross_pct is not None:
        ax.plot(x_vals, success_net_pct, label="mit Gebühren", color=COMPANY_BLUE)
        ax.plot(x_vals, success_gross_pct, label="ohne Gebühren", color="grey")
    else:
        ax.plot(x_vals, success_net_pct, color=COMPANY_BLUE)

    ax.set_xlabel("Entnahmesatz in Prozent p.a.", fontsize=FONT_SIZE_LABEL + 1)
    ax.set_ylabel("Historische Erfolgsquote in %", fontsize=FONT_SIZE_LABEL + 1)

    def percent_formatter_y(y, _pos):
        return f"{y:.0f} %"

    def percent_formatter_x(x, _pos):
        return f"{x:.1f} %"

    ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter_y))
    ax.xaxis.set_major_formatter(FuncFormatter(percent_formatter_x))
    ax.set_ylim(0.0, 105.0)

    ax.tick_params(axis="both", labelsize=FONT_SIZE_TICKS + 1)

    tax_str = f"mit Steuer {tax_rate*100:.2f} %" if tax_rate > 0 else "ohne Steuer"
    compare_str = ", Vergleich ohne Gebühren" if show_gross_line else ""

    extra = (
        f"Entnahmesätze {rate_min*100:.1f}–{rate_max*100:.1f} %, "
        f"{n_cohorts} Läufe, {tax_str}{compare_str}"
    )

    ax.set_title(
        format_chart_title(
            "Erfolgswahrscheinlichkeit versus Entnahmesatz",
            port_cfg_net,
            sim_cfg,
            extra=extra,
        ),
        fontsize=FONT_SIZE_TITLE + 1,
    )

    if show_gross_line and success_gross_pct is not None:
        legend = ax.legend()
        for text in legend.get_texts():
            text.set_fontsize(FONT_SIZE_LEGEND + 1)

    plt.tight_layout()
    return fig


def compute_terminal_wealth_distribution(
    portfolio_returns: pd.Series,
    sim_cfg: SimulationConfig,
    tax_rate: float,
) -> pd.Series:
    """
    Compute terminal wealth across all cohorts for a given configuration.

    The returned Series has the cohort start dates as index.
    """
    periods = sim_cfg.horizon_years * sim_cfg.periods_per_year
    n_cohorts = len(portfolio_returns) - periods + 1

    terminal_wealth = []
    start_dates = []

    for start in range(n_cohorts):
        window = portfolio_returns.iloc[start:start + periods]
        start_dates.append(window.index[0])
        cfg = SimulationConfig(
            annual_withdrawal_rate=sim_cfg.annual_withdrawal_rate,
            initial_wealth=sim_cfg.initial_wealth,
            periods_per_year=sim_cfg.periods_per_year,
            horizon_years=sim_cfg.horizon_years,
        )
        path = simulate_constant_withdrawal(window, cfg, tax_rate=tax_rate)
        terminal_wealth.append(path["wealth"].iloc[-1])

    series = pd.Series(terminal_wealth, index=pd.to_datetime(start_dates))
    series.index.name = "start_date"
    series.attrs["n_cohorts"] = n_cohorts

    return series


def plot_terminal_wealth_hist(
    term_wealth: pd.Series,
    title: str,
    sim_cfg: SimulationConfig,
) -> Figure:
    """
    Plot a histogram of terminal wealth across cohorts, with
    Min/Max/Mittelwert/Median inkl. Jahr-Monat in der Legende.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_16_9, dpi=DPI_EXPORT)
    ax.hist(term_wealth.values, bins=30, color=COMPANY_BLUE, alpha=0.8)

    ax.xaxis.set_major_formatter(FuncFormatter(euro_formatter))

    min_val = term_wealth.min()
    max_val = term_wealth.max()
    mean_val = term_wealth.mean()
    median_val = term_wealth.median()

    start_min = term_wealth.idxmin()
    start_max = term_wealth.idxmax()
    start_median = (term_wealth - median_val).abs().idxmin()

    def fmt_ym(d):
        return d.strftime("%Y-%m") if isinstance(d, pd.Timestamp) else str(d)

    label_min = f"Minimum: {euro_formatter(min_val, None)} ({fmt_ym(start_min)})"
    label_max = f"Maximum: {euro_formatter(max_val, None)} ({fmt_ym(start_max)})"
    label_mean = f"Mittelwert: {euro_formatter(mean_val, None)}"
    label_median = f"Median: {euro_formatter(median_val, None)} ({fmt_ym(start_median)})"

    ax.axvline(min_val, color=WORST_COLOR, linestyle="--", linewidth=1.2, label=label_min)
    ax.axvline(max_val, color=BEST_COLOR, linestyle="--", linewidth=1.2, label=label_max)
    ax.axvline(mean_val, color="grey", linestyle=":", linewidth=1.2, label=label_mean)
    ax.axvline(median_val, color=COMPANY_BLUE, linestyle="-", linewidth=1.5, label=label_median)

    n_cohorts = term_wealth.attrs.get("n_cohorts")
    if n_cohorts is not None:
        ax.text(
            0.99,
            0.02,
            f"{n_cohorts} historische Läufe",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=FONT_SIZE_ANNOT,
        )

    ax.set_title(title)
    ax.set_xlabel(f"Restvermögen nach {sim_cfg.horizon_years} Jahren")
    ax.set_ylabel("Anzahl Kohorten")
    ax.legend(loc="upper right")
    plt.tight_layout()
    return fig


def plot_cumulative_withdrawals(
    path: pd.DataFrame,
    sim_cfg: SimulationConfig,
    title: str,
    inflation: Optional[pd.Series] = None,
    real_mode: bool = True,
) -> Figure:
    """
    Plot the cumulated withdrawals over the retirement period.

    Uses 'withdrawal_net' from the simulated path.

    If inflation is provided and real_mode == True, a second line is plotted
    that shows the same Entnahmen in nominalen (inflationsindexierten) Euro.
    """
    months = np.arange(1, len(path) + 1)
    years = months / sim_cfg.periods_per_year

    cum_withdrawals = path["withdrawal_net"].cumsum()

    fig, ax = plt.subplots(figsize=FIGSIZE_16_9, dpi=DPI_EXPORT)

    label_base = "Kumulierte Entnahmen (real)" if real_mode else "Kumulierte Entnahmen (nominal)"
    ax.plot(
        years,
        cum_withdrawals,
        color=COMPANY_BLUE,
        linewidth=2.0,
        label=label_base,
    )

    total_nominal = None

    if inflation is not None and real_mode:
        infl = inflation.reindex(path.index).fillna(0.0)
        infl_factor = (1.0 + infl).cumprod()
        withdrawals_nominal = path["withdrawal_net"] * infl_factor
        cum_nominal = withdrawals_nominal.cumsum()

        ax.plot(
            years,
            cum_nominal,
            color=BAND_50_COLOR,
            linestyle="--",
            linewidth=2.0,
            label="Kumulierte Entnahmen (nominal, inflationsindexiert)",
        )
        total_nominal = cum_nominal.iloc[-1]

    ax.set_xlabel("Jahre im Ruhestand")
    ax.set_ylabel("Kumulierte Entnahmen")
    set_euro_yaxis(ax)

    if total_nominal is not None:
        text = (
            f"Summe nach {sim_cfg.horizon_years} Jahren (nominal): "
            f"{euro_formatter(total_nominal, None)}"
        )
    else:
        total = cum_withdrawals.iloc[-1]
        suffix = "real" if real_mode else "nominal"
        text = (
            f"Summe nach {sim_cfg.horizon_years} Jahren ({suffix}): "
            f"{euro_formatter(total, None)}"
        )

    ax.text(
        0.99,
        0.02,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=FONT_SIZE_ANNOT,
    )

    ax.legend()
    ax.set_title(title)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# STREAMLIT FRONT-END (VARIANTE B: NUR IN run_app())
# ---------------------------------------------------------------------------

def run_app() -> None:
    import tempfile
    import streamlit as st

    st.set_page_config(page_title="Verrentungs-Simulation (MSCI World + REXP + Gold)", layout="wide")
    st.title("Verrentungs-Simulation: MSCI World, REXP und Gold")
    st.markdown(
        "Diese Anwendung ist für Beratungsgespräche gedacht. "
        "Änderungen werden erst nach Klick auf „Berechnung starten“ übernommen."
    )

    default_file_exists = DATA_FILE.is_file()

    def _euro_str(x: float) -> str:
        return f"{x:,.0f} €".replace(",", ".")

    def _fmt_date_ym(d: pd.Timestamp) -> str:
        if isinstance(d, pd.Timestamp):
            return d.strftime("%Y-%m")
        return str(d)

    def _simulate_path_for_start_date(
        portfolio_returns: pd.Series,
        sim_cfg: SimulationConfig,
        tax_rate: float,
        start_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Simulate a withdrawal path for a specific cohort start date.
        """
        periods = sim_cfg.horizon_years * sim_cfg.periods_per_year
        idx = portfolio_returns.index.get_indexer([pd.to_datetime(start_date)])[0]
        if idx < 0:
            raise ValueError("Startdatum der Kohorte wurde in der Renditereihe nicht gefunden.")
        window = portfolio_returns.iloc[idx:idx + periods]

        cfg = SimulationConfig(
            annual_withdrawal_rate=sim_cfg.annual_withdrawal_rate,
            initial_wealth=sim_cfg.initial_wealth,
            periods_per_year=sim_cfg.periods_per_year,
            horizon_years=sim_cfg.horizon_years,
        )
        return simulate_constant_withdrawal(window, cfg, tax_rate=tax_rate)

    def _init_defaults() -> None:
        """
        Setzt Default-Werte in st.session_state, falls noch nicht vorhanden.
        Dadurch bleibt die Sidebar stabil, und der Reset-Button ist einfach umzusetzen.
        """
        st.session_state.setdefault("erweitert", False)

        if default_file_exists:
            st.session_state.setdefault("datenquelle", "Standardpfad verwenden")
        else:
            st.session_state.setdefault("datenquelle", "Excel-Datei hochladen")

        st.session_state.setdefault("excel_path_input", str(DATA_FILE))
        st.session_state.setdefault("uploaded_file", None)

        st.session_state.setdefault("sheet_name", "Import_Daten")
        st.session_state.setdefault("date_column", "Dates")
        st.session_state.setdefault("cpi_column", "Inflation DE")
        st.session_state.setdefault("col_msci", "NDDUWI Index")
        st.session_state.setdefault("col_rexp", "REXP Index")
        st.session_state.setdefault("col_gold", "Gold")

        st.session_state.setdefault("w_msci_pct", float(PORTFOLIO_WEIGHTS.get("msci_world", 0.60) * 100.0))
        st.session_state.setdefault("w_rexp_pct", float(PORTFOLIO_WEIGHTS.get("rexp", 0.35) * 100.0))
        st.session_state.setdefault("w_gold_pct", float(PORTFOLIO_WEIGHTS.get("Gold", 0.05) * 100.0))

        st.session_state.setdefault("apply_fees", APPLY_FEES)
        st.session_state.setdefault("annual_fee_pct", float(ANNUAL_FEE * 100.0))

        st.session_state.setdefault("apply_tax", APPLY_TAX)
        st.session_state.setdefault("tax_rate_pct", float(CAPITAL_GAINS_TAX_RATE * 100.0))

        st.session_state.setdefault("use_inflation", USE_INFLATION)
        st.session_state.setdefault("withdrawal_rate_pct", float(WITHDRAWAL_RATE * 100.0))
        st.session_state.setdefault("initial_wealth", float(INITIAL_WEALTH))
        st.session_state.setdefault("horizon_years", int(HORIZON_YEARS))

        st.session_state.setdefault("rate_min_pct", float(SUCCESS_RATE_MIN * 100.0))
        st.session_state.setdefault("rate_max_pct", float(SUCCESS_RATE_MAX * 100.0))
        st.session_state.setdefault("rate_step_pp", float(SUCCESS_RATE_STEP * 100.0))
        st.session_state.setdefault("show_gross_line", False)

        st.session_state.setdefault("results", None)
        st.session_state.setdefault("last_config_fingerprint", None)

    def _reset_settings() -> None:
        """
        Setzt alle Eingaben auf die Default-Werte zurück und entfernt gespeicherte Resultate.
        """
        keys_to_clear = [
            "erweitert",
            "datenquelle",
            "excel_path_input",
            "uploaded_file",
            "sheet_name",
            "date_column",
            "cpi_column",
            "col_msci",
            "col_rexp",
            "col_gold",
            "w_msci_pct",
            "w_rexp_pct",
            "w_gold_pct",
            "apply_fees",
            "annual_fee_pct",
            "apply_tax",
            "tax_rate_pct",
            "use_inflation",
            "withdrawal_rate_pct",
            "initial_wealth",
            "horizon_years",
            "rate_min_pct",
            "rate_max_pct",
            "rate_step_pp",
            "show_gross_line",
            "results",
            "last_config_fingerprint",
        ]
        for k in keys_to_clear:
            if k in st.session_state:
                del st.session_state[k]
        _init_defaults()

    def _resolve_excel_path() -> Optional[Path]:
        if st.session_state["datenquelle"] == "Standardpfad verwenden":
            return DATA_FILE if default_file_exists else None
        if st.session_state["datenquelle"] == "Pfad zur Excel-Datei eingeben":
            p = Path(st.session_state["excel_path_input"])
            return p if p.is_file() else None
        return None

    def _load_panel_from_source() -> pd.DataFrame:
        asset_columns = {
            "msci_world": st.session_state["col_msci"],
            "rexp": st.session_state["col_rexp"],
            "Gold": st.session_state["col_gold"],
        }

        datenquelle = st.session_state["datenquelle"]

        if datenquelle in ("Standardpfad verwenden", "Pfad zur Excel-Datei eingeben"):
            excel_path = _resolve_excel_path()
            if excel_path is None:
                raise FileNotFoundError("Excel-Datei wurde nicht gefunden.")
            data_cfg = DataConfig(
                excel_path=excel_path,
                sheet_name=st.session_state["sheet_name"],
                date_column=st.session_state["date_column"],
                cpi_column=st.session_state["cpi_column"],
                asset_columns=asset_columns,
            )
            return load_market_data(data_cfg)

        uploaded_file = st.session_state.get("uploaded_file", None)
        if uploaded_file is None:
            raise ValueError("Keine Excel-Datei hochgeladen.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_path = Path(tmp.name)

        data_cfg = DataConfig(
            excel_path=temp_path,
            sheet_name=st.session_state["sheet_name"],
            date_column=st.session_state["date_column"],
            cpi_column=st.session_state["cpi_column"],
            asset_columns=asset_columns,
        )
        return load_market_data(data_cfg)

    def _current_config_fingerprint() -> tuple:
        """
        Erzeugt einen „Fingerabdruck“ der aktuellen Einstellungen.
        Damit können wir erkennen, ob sich etwas geändert hat, ohne sofort neu zu rechnen.
        """
        uploaded = st.session_state.get("uploaded_file", None)
        uploaded_sig = None
        if uploaded is not None:
            try:
                uploaded_sig = (uploaded.name, uploaded.size)
            except Exception:
                uploaded_sig = "uploaded"

        return (
            st.session_state.get("datenquelle"),
            st.session_state.get("excel_path_input"),
            uploaded_sig,
            st.session_state.get("sheet_name"),
            st.session_state.get("date_column"),
            st.session_state.get("cpi_column"),
            st.session_state.get("col_msci"),
            st.session_state.get("col_rexp"),
            st.session_state.get("col_gold"),
            float(st.session_state.get("w_msci_pct")),
            float(st.session_state.get("w_rexp_pct")),
            float(st.session_state.get("w_gold_pct")),
            bool(st.session_state.get("apply_fees")),
            float(st.session_state.get("annual_fee_pct")),
            bool(st.session_state.get("apply_tax")),
            float(st.session_state.get("tax_rate_pct")),
            bool(st.session_state.get("use_inflation")),
            float(st.session_state.get("withdrawal_rate_pct")),
            float(st.session_state.get("initial_wealth")),
            int(st.session_state.get("horizon_years")),
            float(st.session_state.get("rate_min_pct")),
            float(st.session_state.get("rate_max_pct")),
            float(st.session_state.get("rate_step_pp")),
            bool(st.session_state.get("show_gross_line")),
        )

    _init_defaults()

    # ---------------------------------------------------------------------------
    # SIDEBAR: BERATER-FREUNDLICHE EINSTELLUNGEN
    # ---------------------------------------------------------------------------

    st.sidebar.header("Einstellungen")

    c_btn1, c_btn2 = st.sidebar.columns(2)
    with c_btn1:
        run_clicked = st.button("Berechnung starten", type="primary")
    with c_btn2:
        if st.button("Zurücksetzen"):
            _reset_settings()
            st.rerun()

    st.sidebar.markdown("")

    st.session_state["erweitert"] = st.sidebar.checkbox(
        "Erweiterte Einstellungen anzeigen",
        value=st.session_state["erweitert"],
        help="Wenn deaktiviert, werden nur die wichtigsten Parameter angezeigt.",
    )

    if default_file_exists:
        daten_optionen = ("Standardpfad verwenden", "Excel-Datei hochladen")
    else:
        daten_optionen = ("Excel-Datei hochladen",)

    if st.session_state["erweitert"]:
        daten_optionen = daten_optionen + ("Pfad zur Excel-Datei eingeben",)

    st.session_state["datenquelle"] = st.sidebar.radio(
        "Datenquelle",
        daten_optionen,
        index=list(daten_optionen).index(st.session_state["datenquelle"])
        if st.session_state["datenquelle"] in daten_optionen
        else 0,
    )

    if st.session_state["datenquelle"] == "Standardpfad verwenden":
        st.sidebar.caption(f"Verwendeter Pfad: {DATA_FILE}")
        if not default_file_exists:
            st.sidebar.warning("Standardpfad nicht gefunden. Bitte Datei hochladen oder Pfad eingeben.")
    elif st.session_state["datenquelle"] == "Pfad zur Excel-Datei eingeben":
        st.session_state["excel_path_input"] = st.sidebar.text_input(
            "Excel-Pfad",
            value=st.session_state["excel_path_input"],
        )
    else:
        st.session_state["uploaded_file"] = st.sidebar.file_uploader("Excel-Datei auswählen", type=["xlsx", "xls"])

    if st.session_state["erweitert"]:
        with st.sidebar.expander("Datenstruktur (Sheet und Spalten)", expanded=False):
            st.session_state["sheet_name"] = st.text_input("Sheet-Name", value=st.session_state["sheet_name"])
            st.session_state["date_column"] = st.text_input("Datums-Spalte", value=st.session_state["date_column"])
            st.session_state["cpi_column"] = st.text_input("Inflations-Spalte (CPI)", value=st.session_state["cpi_column"])
            st.session_state["col_msci"] = st.text_input("MSCI World Spalte", value=st.session_state["col_msci"])
            st.session_state["col_rexp"] = st.text_input("REXP Spalte", value=st.session_state["col_rexp"])
            st.session_state["col_gold"] = st.text_input("Gold Spalte", value=st.session_state["col_gold"])

    st.sidebar.divider()
    st.sidebar.subheader("Portfolio")

    # -----------------------------------------------------------------------
    # GEWICHTE: JEDER REGLER 0–100 %
    #
    # Berater-UX:
    # Jeder Regler darf bis 100 % gehen (intuitiv).
    # Wenn die Summe > 100 % ist, zeigen wir einen Hinweis und blockieren die Berechnung.
    # Wenn die Summe < 100 % ist, wird der Rest automatisch als Liquidität ergänzt.
    # -----------------------------------------------------------------------

    st.session_state["w_msci_pct"] = st.sidebar.slider(
        "MSCI World (%)",
        0.0,
        100.0,
        float(st.session_state["w_msci_pct"]),
        1.0,
    )

    st.session_state["w_rexp_pct"] = st.sidebar.slider(
        "REXP (%)",
        0.0,
        100.0,
        float(st.session_state["w_rexp_pct"]),
        1.0,
    )

    st.session_state["w_gold_pct"] = st.sidebar.slider(
        "Gold (%)",
        0.0,
        100.0,
        float(st.session_state["w_gold_pct"]),
        1.0,
    )

    sum_weights_pct = float(
        st.session_state["w_msci_pct"]
        + st.session_state["w_rexp_pct"]
        + st.session_state["w_gold_pct"]
    )

    rest_cash_pct = float(max(0.0, 100.0 - sum_weights_pct))

    m1, m2 = st.sidebar.columns(2)
    with m1:
        st.metric("Summe", f"{sum_weights_pct:.0f} %")
    with m2:
        st.metric("Liquidität", f"{rest_cash_pct:.0f} %")

    if sum_weights_pct > 100.0 + 1e-9:
        st.sidebar.warning(
            f"Achtung: Die Summe der Gewichte beträgt {sum_weights_pct:.1f} % und ist größer als 100 %.\n"
            "Bitte Gewichte reduzieren. Die Berechnung wird sonst nicht gestartet."
        )
    else:
        st.sidebar.caption("Wenn die Summe unter 100 % liegt, wird der Rest automatisch als Liquidität ergänzt.")

    st.sidebar.divider()
    st.sidebar.subheader("Kosten und Entnahme")

    st.session_state["apply_fees"] = st.sidebar.checkbox("Gebühren berücksichtigen", value=st.session_state["apply_fees"])
    st.session_state["annual_fee_pct"] = st.sidebar.number_input(
        "Gebühr pro Jahr (%)",
        min_value=0.0,
        max_value=10.0,
        value=float(st.session_state["annual_fee_pct"]),
        step=0.01,
        format="%.2f",
        disabled=not st.session_state["apply_fees"],
    )

    st.session_state["apply_tax"] = st.sidebar.checkbox("Steuer berücksichtigen", value=st.session_state["apply_tax"])
    st.session_state["tax_rate_pct"] = st.sidebar.number_input(
        "Steuersatz (%)",
        min_value=0.0,
        max_value=50.0,
        value=float(st.session_state["tax_rate_pct"]),
        step=0.10,
        format="%.2f",
        disabled=not st.session_state["apply_tax"],
    )

    st.session_state["use_inflation"] = st.sidebar.radio(
        "Rechnungsmodus",
        ("Inflationsbereinigt (real)", "Nominal"),
        index=0 if st.session_state["use_inflation"] else 1,
    ) == "Inflationsbereinigt (real)"

    st.session_state["withdrawal_rate_pct"] = st.sidebar.slider(
        "Entnahmesatz p.a. (%)",
        1.0,
        10.0,
        float(st.session_state["withdrawal_rate_pct"]),
        0.1,
        format="%.1f",
    )

    st.session_state["initial_wealth"] = st.sidebar.number_input(
        "Startvermögen (€)",
        min_value=0.0,
        max_value=100_000_000.0,
        value=float(st.session_state["initial_wealth"]),
        step=10_000.0,
        format="%.0f",
    )

    st.session_state["horizon_years"] = st.sidebar.slider(
        "Horizont (Jahre)",
        5,
        60,
        int(st.session_state["horizon_years"]),
        1,
    )

    if st.session_state["erweitert"]:
        st.sidebar.divider()
        st.sidebar.subheader("Erfolgskurve (optional)")
        st.session_state["rate_min_pct"], st.session_state["rate_max_pct"] = st.sidebar.slider(
            "Spannweite (% p.a.)",
            1.0,
            10.0,
            (float(st.session_state["rate_min_pct"]), float(st.session_state["rate_max_pct"])),
            0.25,
            format="%.2f",
        )
        st.session_state["rate_step_pp"] = st.sidebar.number_input(
            "Schritt (Prozentpunkte)",
            min_value=0.05,
            max_value=2.0,
            value=float(st.session_state["rate_step_pp"]),
            step=0.05,
            format="%.2f",
        )
        st.session_state["show_gross_line"] = st.sidebar.checkbox(
            "Vergleich ohne Gebühren anzeigen",
            value=st.session_state["show_gross_line"],
        )

    # ---------------------------------------------------------------------------
    # MAIN: RECHNUNG NUR BEI BUTTON-KLICK
    # ---------------------------------------------------------------------------

    current_fp = _current_config_fingerprint()

    if st.session_state["results"] is None:
        st.info("Bitte Einstellungen links wählen und dann „Berechnung starten“ klicken.")
    else:
        if st.session_state.get("last_config_fingerprint") != current_fp:
            st.warning("Einstellungen wurden geändert. Bitte „Berechnung starten“ klicken, um die Ergebnisse zu aktualisieren.")

    if run_clicked:
        sum_weights_pct_run = float(
            st.session_state["w_msci_pct"]
            + st.session_state["w_rexp_pct"]
            + st.session_state["w_gold_pct"]
        )

        if sum_weights_pct_run > 100.0 + 1e-9:
            st.error(
                f"Die Summe der Portfolio-Gewichte beträgt {sum_weights_pct_run:.1f} % und überschreitet 100 %.\n\n"
                "Bitte reduzieren Sie die Gewichte so, dass die Summe maximal 100 % ist."
            )
            st.stop()

        rest_cash_pct_run = float(max(0.0, 100.0 - sum_weights_pct_run))

        rate_min = float(st.session_state["rate_min_pct"] / 100.0)
        rate_max = float(st.session_state["rate_max_pct"] / 100.0)
        rate_step = float(st.session_state["rate_step_pp"] / 100.0)

        if rate_min >= rate_max:
            st.error("Für die Erfolgskurve muss die minimale Rate kleiner als die maximale Rate sein.")
            st.stop()

        weights = {
            "msci_world": float(st.session_state["w_msci_pct"] / 100.0),
            "rexp": float(st.session_state["w_rexp_pct"] / 100.0),
            "Gold": float(st.session_state["w_gold_pct"] / 100.0),
        }
        if rest_cash_pct_run > 1e-9:
            weights["cash"] = float(rest_cash_pct_run / 100.0)

        annual_fee = float(st.session_state["annual_fee_pct"] / 100.0) if st.session_state["apply_fees"] else 0.0
        tax_rate = float(st.session_state["tax_rate_pct"] / 100.0) if st.session_state["apply_tax"] else 0.0
        withdrawal_rate = float(st.session_state["withdrawal_rate_pct"] / 100.0)

        sim_cfg = SimulationConfig(
            annual_withdrawal_rate=withdrawal_rate,
            initial_wealth=float(st.session_state["initial_wealth"]),
            periods_per_year=PERIODS_PER_YEAR,
            horizon_years=int(st.session_state["horizon_years"]),
        )

        port_cfg_net = PortfolioConfig(
            name="portfolio_net",
            weights=weights,
            annual_fee=annual_fee,
            use_fees=bool(st.session_state["apply_fees"]),
            use_inflation=bool(st.session_state["use_inflation"]),
        )

        port_cfg_gross = PortfolioConfig(
            name="portfolio_gross",
            weights=weights,
            annual_fee=0.0,
            use_fees=False,
            use_inflation=bool(st.session_state["use_inflation"]),
        )

        with st.spinner("Berechnung läuft..."):
            try:
                panel = _load_panel_from_source()
            except Exception as e:
                st.error(f"Fehler beim Laden der Marktdaten: {e}")
                st.stop()

            nominal_rets = compute_nominal_returns(panel)
            inflation_series = nominal_rets["inflation"]

            portfolio_returns_net = prepare_portfolio_returns(panel, port_cfg_net)
            portfolio_returns_gross = prepare_portfolio_returns(panel, port_cfg_gross)

            periods_needed = sim_cfg.horizon_years * sim_cfg.periods_per_year
            if len(portfolio_returns_net) < periods_needed:
                st.error(
                    f"Nicht genügend Daten für den gewählten Horizont. "
                    f"Benötigt: {periods_needed} Monate, verfügbar: {len(portfolio_returns_net)}."
                )
                st.stop()

            tax_str = f"mit Steuer {tax_rate*100:.2f} %" if tax_rate > 0 else "ohne Steuer"

            matrix = build_wealth_matrix(portfolio_returns_net, sim_cfg, tax_rate=tax_rate)
            n_cohorts = matrix.attrs.get("n_cohorts", matrix.shape[1])

            cohort_summary = summarise_cohorts(matrix)
            term_wealth = compute_terminal_wealth_distribution(portfolio_returns_net, sim_cfg, tax_rate=tax_rate)

            terminals = matrix.iloc[-1]
            best_start = terminals.idxmax()
            worst_start = terminals.idxmin()
            median_start = (terminals - terminals.median()).abs().idxmin()

            path_median = _simulate_path_for_start_date(portfolio_returns_net, sim_cfg, tax_rate, median_start)
            path_best = _simulate_path_for_start_date(portfolio_returns_net, sim_cfg, tax_rate, best_start)
            path_worst = _simulate_path_for_start_date(portfolio_returns_net, sim_cfg, tax_rate, worst_start)

            erfolg_aktuell = float(((matrix > 0).all(axis=0)).mean())

            st.session_state["results"] = {
                "panel": panel,
                "inflation_series": inflation_series,
                "sim_cfg": sim_cfg,
                "port_cfg_net": port_cfg_net,
                "portfolio_returns_net": portfolio_returns_net,
                "portfolio_returns_gross": portfolio_returns_gross,
                "tax_rate": tax_rate,
                "tax_str": tax_str,
                "matrix": matrix,
                "n_cohorts": n_cohorts,
                "cohort_summary": cohort_summary,
                "term_wealth": term_wealth,
                "best_start": best_start,
                "worst_start": worst_start,
                "median_start": median_start,
                "path_median": path_median,
                "path_best": path_best,
                "path_worst": path_worst,
                "erfolg_aktuell": erfolg_aktuell,
                "rate_min": rate_min,
                "rate_max": rate_max,
                "rate_step": rate_step,
                "show_gross_line": bool(st.session_state.get("show_gross_line", False)),
            }

            st.session_state["last_config_fingerprint"] = current_fp

        st.rerun()

    # ---------------------------------------------------------------------------
    # OUTPUT (wenn Ergebnisse vorhanden)
    # ---------------------------------------------------------------------------

    if st.session_state["results"] is not None:
        r = st.session_state["results"]

        st.subheader("Übersicht")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Startvermögen", _euro_str(r["sim_cfg"].initial_wealth))
        with col2:
            st.metric("Entnahmesatz", f"{r['sim_cfg'].annual_withdrawal_rate*100:.1f} % p.a.")
        with col3:
            st.metric("Horizont", f"{r['sim_cfg'].horizon_years} Jahre")
        with col4:
            st.metric("Historische Läufe", f"{r['n_cohorts']}")

        st.markdown(f"Historische Erfolgsquote für den gewählten Entnahmesatz: **{r['erfolg_aktuell']*100:.1f} %**")
        st.markdown(f"Portfolio-Mix: **{format_weights(r['port_cfg_net'].weights)}**")

        tabs = st.tabs(
            [
                "1. Historische Bandbreite",
                "2. Kumulierte Entnahmen",
                "3. Erfolgskurve",
                "4. Kohorten und Export",
            ]
        )

        with tabs[0]:
            st.header("Historische Bandbreite (Fan-Chart)")

            extra_fan = r["tax_str"]
            extra_fan += f", {r['n_cohorts']} Läufe"

            title_fan = format_chart_title(
                "Historische Bandbreite",
                r["port_cfg_net"],
                r["sim_cfg"],
                extra=extra_fan,
            )

            fig_fan = plot_fan_chart(r["matrix"], title=title_fan)
            st.pyplot(fig_fan, use_container_width=True)
            plt.close(fig_fan)

            with st.expander("Optional: Alle historischen Vermögenspfade anzeigen", expanded=False):
                title_paths = format_chart_title(
                    "Historische Vermögenspfade aller Kohorten",
                    r["port_cfg_net"],
                    r["sim_cfg"],
                    extra=extra_fan,
                )
                fig_paths = plot_all_wealth_paths(r["matrix"], title=title_paths)
                st.pyplot(fig_paths, use_container_width=True)
                plt.close(fig_paths)

        with tabs[1]:
            st.header("Kumulierte Entnahmen")

            optionen = ("Median-Kohorte", "Beste Kohorte", "Schlechteste Kohorte")
            auswahl = st.selectbox("Beispielkohorte auswählen", optionen, index=0)

            if auswahl == "Beste Kohorte":
                path = r["path_best"]
                start_date = r["best_start"]
            elif auswahl == "Schlechteste Kohorte":
                path = r["path_worst"]
                start_date = r["worst_start"]
            else:
                path = r["path_median"]
                start_date = r["median_start"]

            st.caption(f"Beispiel-Startdatum: {_fmt_date_ym(pd.to_datetime(start_date))}")

            title_cum = format_chart_title(
                "Kumulierte Entnahmen",
                r["port_cfg_net"],
                r["sim_cfg"],
                extra=r["tax_str"],
            )
            infl_for_plot = r["inflation_series"] if r["port_cfg_net"].use_inflation else None

            fig_cum = plot_cumulative_withdrawals(
                path,
                r["sim_cfg"],
                title_cum,
                inflation=infl_for_plot,
                real_mode=r["port_cfg_net"].use_inflation,
            )
            st.pyplot(fig_cum, use_container_width=True)
            plt.close(fig_cum)

        with tabs[2]:
            st.header("Erfolgskurve")

            with st.spinner("Erfolgskurve wird berechnet..."):
                fig_success = plot_success_curve(
                    r["portfolio_returns_gross"],
                    r["portfolio_returns_net"],
                    r["sim_cfg"],
                    r["port_cfg_net"],
                    rate_min=r["rate_min"],
                    rate_max=r["rate_max"],
                    rate_step=r["rate_step"],
                    tax_rate=r["tax_rate"],
                    show_gross_line=r["show_gross_line"],
                )
            st.pyplot(fig_success, use_container_width=True)
            plt.close(fig_success)

        with tabs[3]:
            st.header("Kohorten und Export")

            st.subheader("Kohortenübersicht (nach Endvermögen sortiert)")
            st.dataframe(r["cohort_summary"], use_container_width=True)

            st.subheader("Downloads")
            wealth_paths_csv = r["matrix"].to_csv(index=True).encode("utf-8-sig")
            cohort_summary_csv = r["cohort_summary"].to_csv(index=False).encode("utf-8-sig")

            col_a, col_b = st.columns(2)
            with col_a:
                st.download_button(
                    "Wealth Paths (alle Kohorten) als CSV herunterladen",
                    data=wealth_paths_csv,
                    file_name="wealth_paths_all_cohorts.csv",
                    mime="text/csv",
                )
            with col_b:
                st.download_button(
                    "Kohortenübersicht als CSV herunterladen",
                    data=cohort_summary_csv,
                    file_name="cohort_summary.csv",
                    mime="text/csv",
                )

            with st.expander("Optional: Verteilung des Restvermögens anzeigen", expanded=False):
                extra_hist = r["tax_str"]
                n_cohorts_tw = r["term_wealth"].attrs.get("n_cohorts", None)
                if n_cohorts_tw is not None:
                    extra_hist += f", {n_cohorts_tw} Läufe"

                title_hist = format_chart_title(
                    f"Verteilung des Restvermögens nach {r['sim_cfg'].horizon_years} Jahren",
                    r["port_cfg_net"],
                    r["sim_cfg"],
                    extra=extra_hist,
                )
                fig_hist = plot_terminal_wealth_hist(r["term_wealth"], title=title_hist, sim_cfg=r["sim_cfg"])
                st.pyplot(fig_hist, use_container_width=True)
                plt.close(fig_hist)


if __name__ == "__main__":
    run_app()

