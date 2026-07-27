"""
app.py
------
Streamlit front-end for the educational portfolio backtesting dashboard.

This file is intentionally "thin": it collects user inputs, calls into
data.py / backtest.py / metrics.py for all the real work, and renders the
results. No price math or statistics live here. AI-powered plain-language
analysis (via a local Ollama model) is handled by ai.py.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ai import analyze_portfolio
from backtest import REBALANCE_FREQUENCIES, run_backtest
from data import TICKERS, get_asset_prices, get_backtest_window
from metrics import annual_returns, drawdown_series, summarize

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Portfolio Allocation Explorer",
    page_icon="📊",
    layout="wide",
)

ASSET_COLORS = {"Stocks": "#3B82F6", "Bonds": "#10B981", "Gold": "#F59E0B"}

# Allocation sliders move in 5-percentage-point increments, so every
# preset below is defined in multiples of 5 to stay reachable via the UI.
SLIDER_STEP = 5

DEFAULT_QUESTION = "Explain this portfolio's risk and return characteristics."

# CSS that pins the AI chat trigger button to a fixed spot on the middle
# right of the viewport, styled as a small circular floating action button.
# `.st-key-ai_fab_button` is the wrapper Streamlit generates automatically
# for any widget created with `key="ai_fab_button"`.
AI_FAB_CSS = """
<style>
.st-key-ai_fab_button button {
    position: fixed;
    top: 50%;
    right: 22px;
    transform: translateY(-50%);
    z-index: 9999;
    width: 56px;
    height: 56px;
    border-radius: 50%;
    font-size: 1.4rem;
    line-height: 1;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.28);
    display: flex;
    align-items: center;
    justify-content: center;
}
</style>
"""


# --------------------------------------------------------------------------
# Formatting helpers (shared by metric cards + the AI summary payload)
# --------------------------------------------------------------------------
def fmt_pct(x) -> str:
    return "N/A" if pd.isna(x) else f"{x * 100:.1f}%"


def fmt_ratio(x) -> str:
    return "N/A" if pd.isna(x) else f"{x:.2f}"


# --------------------------------------------------------------------------
# Common portfolio models (educational presets)
# --------------------------------------------------------------------------
# Three illustrative allocations built only from the three assets this
# dashboard supports. These are generic, textbook-style risk profiles for
# educational comparison -- not personalized recommendations.
PORTFOLIO_MODELS = {
    "Conservative": {
        "label": "🛡️ Conservative",
        "weights": {"Stocks": 30, "Bonds": 60, "Gold": 10},
        "tagline": "Capital preservation first",
        "description": (
            "Leans heavily on bonds to dampen volatility and drawdowns, with a modest "
            "stock allocation for some growth and a small gold position for further "
            "diversification. Historically trades away upside for a smoother ride."
        ),
    },
    "Moderate": {
        "label": "⚖️ Moderate",
        "weights": {"Stocks": 60, "Bonds": 30, "Gold": 10},
        "tagline": "Balanced growth and stability",
        "description": (
            "A classic 60/40 stock/bond split trimmed to make room for a 10% gold "
            "position. Aims to balance long-run growth against meaningful downside "
            "protection -- a common educational reference point for a 'balanced' portfolio."
        ),
    },
    "Aggressive": {
        "label": "🚀 Aggressive",
        "weights": {"Stocks": 85, "Bonds": 10, "Gold": 5},
        "tagline": "Growth-focused, higher volatility",
        "description": (
            "Concentrates in stocks to maximize long-run growth potential, with only a "
            "small ballast of bonds and gold. Historically comes with higher volatility "
            "and deeper drawdowns along the way."
        ),
    },
}


def _apply_preset(weights: dict) -> None:
    """
    Callback used by the preset buttons: writes directly into the slider
    widgets' session_state keys. Streamlit runs callbacks before the script
    reruns, so the sliders will reflect these values on the very next render.
    """
    st.session_state["stocks_pct"] = weights["Stocks"]
    st.session_state["bonds_pct"] = weights["Bonds"]
    st.session_state["gold_pct"] = weights["Gold"]


# --------------------------------------------------------------------------
# Sidebar: user inputs
# --------------------------------------------------------------------------
def render_sidebar() -> dict:
    """
    Render all sidebar controls and return the collected inputs as a dict.

    Each of Stocks / Bonds / Gold is an independent slider (no
    auto-calculation), moving in 5-percentage-point steps. Three small
    preset buttons sit directly underneath the sliders for quick
    Conservative / Moderate / Aggressive starting points. The running
    total is shown in red until it equals exactly 100%, at which point it
    turns green -- and only then is the "Run Backtest" button enabled.
    """
    st.sidebar.header("Portfolio Allocation")
    st.sidebar.caption(
        "Set each allocation independently, in steps of 5%. The total must "
        "equal exactly 100% before you can run the backtest."
    )

    # Seed each slider's session_state default exactly once. After this,
    # sliders are created with `key=` only (no `value=`), which is the
    # pattern Streamlit expects when a value may later be set programmatically
    # (e.g. by the preset buttons below) -- passing both `value=` and `key=`
    # on every rerun triggers a harmless but noisy Streamlit warning.
    for key, default in (("stocks_pct", 60), ("bonds_pct", 30), ("gold_pct", 10)):
        if key not in st.session_state:
            st.session_state[key] = default

    stocks_pct = st.sidebar.slider(
        f"Stocks allocation (%) — {TICKERS['Stocks']}",
        min_value=0, max_value=100, step=SLIDER_STEP, key="stocks_pct",
    )
    bonds_pct = st.sidebar.slider(
        f"Bonds allocation (%) — {TICKERS['Bonds']}",
        min_value=0, max_value=100, step=SLIDER_STEP, key="bonds_pct",
    )
    gold_pct = st.sidebar.slider(
        f"Gold allocation (%) — {TICKERS['Gold']}",
        min_value=0, max_value=100, step=SLIDER_STEP, key="gold_pct",
    )

    # Three small preset buttons directly underneath the sliders.
    st.sidebar.caption("Quick presets:")
    preset_cols = st.sidebar.columns(3)
    for col, (name, model) in zip(preset_cols, PORTFOLIO_MODELS.items()):
        weights = model["weights"]
        tooltip = (
            f"{model['tagline']} — Stocks {weights['Stocks']}% / "
            f"Bonds {weights['Bonds']}% / Gold {weights['Gold']}%. "
            f"{model['description']}"
        )
        col.button(
            model["label"],
            key=f"preset_btn_{name}",
            help=tooltip,
            on_click=_apply_preset,
            args=(weights,),
            width="stretch",
        )

    total_pct = stocks_pct + bonds_pct + gold_pct
    is_valid_total = total_pct == 100
    color = "#16A34A" if is_valid_total else "#DC2626"  # green / red
    status_icon = "✅" if is_valid_total else "⚠️"

    st.sidebar.markdown(
        f"""
        <div style="padding: 0.6rem 0.8rem; border-radius: 0.5rem;
                    background-color: {color}1A; border: 1px solid {color};
                    margin-top: 0.5rem;">
            <span style="font-size: 1.05rem; font-weight: 600; color: {color};">
                {status_icon} Total Allocation: {total_pct}%
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not is_valid_total:
        st.sidebar.caption(
            f"Adjust the sliders so the total is exactly 100% "
            f"(currently {'over' if total_pct > 100 else 'under'} by "
            f"{abs(100 - total_pct)} percentage point(s))."
        )

    st.sidebar.divider()
    st.sidebar.header("Backtest Settings")

    initial_investment = st.sidebar.number_input(
        "Initial investment ($)",
        min_value=100,
        max_value=100_000_000,
        value=10_000,
        step=500,
    )

    rebalance_frequency = st.sidebar.selectbox(
        "Rebalancing frequency",
        options=list(REBALANCE_FREQUENCIES.keys()),
        index=0,
        help=(
            "Buy & Hold: weights drift with the market and are never reset. "
            "Monthly/Quarterly/Annual: the portfolio is reset back to target "
            "weights at the start of each period."
        ),
    )

    st.sidebar.caption(
        "The backtest always uses the longest history available across all "
        "three tickers (see 'Backtesting Window' at the bottom of the page)."
    )

    run_clicked = st.sidebar.button(
        "Run Backtest",
        type="primary",
        width="stretch",
        disabled=not is_valid_total,
    )

    return {
        "weights": {"Stocks": stocks_pct / 100, "Bonds": bonds_pct / 100, "Gold": gold_pct / 100},
        "total_pct": total_pct,
        "initial_investment": float(initial_investment),
        "rebalance_frequency": rebalance_frequency,
        "run_clicked": run_clicked,
    }


def validate_inputs(inputs: dict) -> list[str]:
    """Return a list of human-readable validation error messages, if any."""
    errors = []

    if inputs["total_pct"] != 100:
        errors.append(f"Allocations must sum to exactly 100% (currently {inputs['total_pct']}%).")

    if inputs["initial_investment"] <= 0:
        errors.append("Initial investment must be greater than zero.")

    return errors


# --------------------------------------------------------------------------
# Chart builders
# --------------------------------------------------------------------------
def build_growth_chart(portfolio: pd.DataFrame) -> go.Figure:
    """Line chart of total portfolio value over time."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=portfolio.index,
            y=portfolio["Total"],
            mode="lines",
            name="Portfolio Value",
            line=dict(color="#3B82F6", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(59, 130, 246, 0.08)",
            hovertemplate="%{x|%b %d, %Y}<br>$%{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Portfolio Growth Over Time",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        height=440,
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
    )
    fig.update_yaxes(tickprefix="$", separatethousands=True)
    return fig


def build_drawdown_chart(portfolio: pd.DataFrame) -> go.Figure:
    """Area chart of drawdown (%) beneath the equity curve."""
    dd = drawdown_series(portfolio["Total"]) * 100
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd,
            mode="lines",
            name="Drawdown",
            line=dict(color="#EF4444", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(239, 68, 68, 0.15)",
            hovertemplate="%{x|%b %d, %Y}<br>%{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="Drawdown Over Time",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_white",
        height=280,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def build_allocation_pie(weights: dict, title: str = "Target Allocation") -> go.Figure:
    """Pie chart of a given target allocation (weights as fractions of 1)."""
    labels = list(weights.keys())
    values = [w * 100 for w in weights.values()]
    colors = [ASSET_COLORS[label] for label in labels]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.45,
                marker=dict(colors=colors),
                textinfo="label+percent",
                hovertemplate="%{label}: %{value:.0f}%<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=350,
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False,
    )
    return fig


def build_annual_returns_table(portfolio: pd.DataFrame) -> pd.DataFrame:
    """Format the annual returns Series into a display-ready DataFrame."""
    yearly = annual_returns(portfolio["Total"])
    df = yearly.reset_index()
    df.columns = ["Year", "Return"]
    df["Return"] = df["Return"].apply(lambda x: f"{x * 100:+.1f}%")
    return df


# --------------------------------------------------------------------------
# Metric cards
# --------------------------------------------------------------------------
def render_metric_cards(stats: dict) -> None:
    """Render the 8 performance metric cards in a responsive grid."""
    row1 = st.columns(4)
    row1[0].metric("CAGR", fmt_pct(stats["cagr"]), help="Compound Annual Growth Rate")
    row1[1].metric("Annualized Volatility", fmt_pct(stats["volatility"]), help="Standard deviation of returns, annualized")
    row1[2].metric("Sharpe Ratio", fmt_ratio(stats["sharpe"]), help="Return per unit of total risk (0% risk-free rate assumed)")
    row1[3].metric("Sortino Ratio", fmt_ratio(stats["sortino"]), help="Return per unit of downside risk only")

    row2 = st.columns(4)
    row2[0].metric("Maximum Drawdown", fmt_pct(stats["max_drawdown"]), help="Largest peak-to-trough decline")
    row2[1].metric("Calmar Ratio", fmt_ratio(stats["calmar"]), help="CAGR divided by maximum drawdown magnitude")
    row2[2].metric("Best Year", fmt_pct(stats["best_year"]), help="Best calendar-year total return")
    row2[3].metric("Worst Year", fmt_pct(stats["worst_year"]), help="Worst calendar-year total return")


# --------------------------------------------------------------------------
# Backtesting window section
# --------------------------------------------------------------------------
def render_backtest_window(window: dict) -> None:
    """
    Display the fixed backtesting window (always the longest history
    available across all three tickers). Shown at the very end of the
    results flow, after the charts and metrics.
    """
    st.subheader("Backtesting Window")
    cols = st.columns(3)
    cols[0].metric("Start Date", window["start"].strftime("%b %d, %Y"))
    cols[1].metric("End Date", window["end"].strftime("%b %d, %Y"))
    cols[2].metric("Span", f"{window['years']:.1f} years")
    st.caption(
        "This dashboard always uses the longest history available across "
        f"all three tickers ({TICKERS['Stocks']}, {TICKERS['Bonds']}, "
        f"{TICKERS['Gold']}). The window is capped by whichever instrument "
        "has the shortest trading history -- typically the gold futures "
        "contract, which has usable daily data on Yahoo Finance from "
        "around 2000 onward."
    )


# --------------------------------------------------------------------------
# Information section (educational reference material)
# --------------------------------------------------------------------------
def render_information_section() -> None:
    """
    Educational reference material explaining the specific assets, tickers,
    methodology, and time period used by the backtest -- kept in a
    collapsed expander so it doesn't crowd the main workflow.
    """
    with st.expander("ℹ️ Information: Assets, Methodology & Time Period", expanded=False):
        st.markdown(
            "This dashboard is **for educational purposes only**. It is **not** a "
            "portfolio optimizer and does **not** provide investment advice or "
            "recommendations. It uses historical data to illustrate general "
            "diversification concepts -- see the tabs below for details."
        )

        tab_assets, tab_methodology, tab_metrics, tab_ai = st.tabs(
            ["Assets & Tickers", "Methodology", "Metrics Glossary", "AI Analysis"]
        )

        with tab_assets:
            st.markdown(
                """
| Asset Class | Ticker | Instrument | What It Represents |
|---|---|---|---|
| Stocks | `SPY` | SPDR S&P 500 ETF Trust | Large-cap U.S. equities; tracks the S&P 500 index |
| Bonds | `VBTIX` | Vanguard Total Bond Market Index Fund, Institutional Shares | Broad U.S. investment-grade bond market (tracks a Bloomberg U.S. Aggregate-style index) |
| Gold | `GC=F` | COMEX Gold Futures (continuous contract) | Gold price via the front-month futures contract, continuously rolled forward |

**Why futures for gold instead of a gold ETF?** `GC=F` gives the longest available daily
history of the three common gold-tracking options, extending back to roughly 2000 on
Yahoo Finance. Its price can differ slightly from spot gold or a gold ETF (like GLD) due
to futures-specific factors such as contract roll timing and storage/carry costs, but it
closely tracks gold's overall price movement.
                """
            )

        with tab_methodology:
            st.markdown(
                """
**Data source:** All prices come from Yahoo Finance via the `yfinance` library, using
adjusted close prices (`auto_adjust=True`), which account for stock splits and dividend/
distribution payments for `SPY` and `VBTIX`.

**Backtesting window:** The dashboard automatically uses the longest *overlapping*
history across all three tickers -- there is no manual date selection. In practice this
window is bounded by whichever instrument has the shortest history, which is typically
`GC=F` (gold futures data on Yahoo Finance generally begins around 2000).

**Data cleaning:** Raw price feeds -- especially futures contracts like `GC=F` -- can
contain occasional data artifacts. Before backtesting, the dashboard:
1. Removes non-positive prices (zero or negative values are never valid).
2. Flags single-day moves larger than 40% as probable bad ticks rather than genuine
   market moves, and smooths them out.
3. Forward-fills any remaining gaps (e.g. a holiday where one instrument trades but
   another doesn't).
4. Trims the dataset to the window where *all three* assets have clean, valid data.

**Return calculation:** Each asset's daily simple return is computed from adjusted close
prices. The portfolio's dollar value in each asset compounds independently day by day,
and the sum across all three assets is the total portfolio value.

**Rebalancing:**
- *Buy & Hold* -- weights drift naturally with market performance and are never reset.
- *Monthly / Quarterly / Annual* -- at the start of each new period, the portfolio is
  sold back to exactly the target weights, realizing gains from outperforming assets
  and reinvesting into underperforming ones.
                """
            )

        with tab_metrics:
            st.markdown(
                """
| Metric | Definition |
|---|---|
| **CAGR** | Compound Annual Growth Rate -- the constant annual growth rate that would take the starting value to the ending value over the full period. |
| **Annualized Volatility** | Standard deviation of daily returns, scaled to an annual figure (× √252 trading days). |
| **Sharpe Ratio** | Annualized return divided by annualized volatility (assumes a 0% risk-free rate, kept simple for comparability). Higher means more return per unit of total risk. |
| **Sortino Ratio** | Like Sharpe, but only penalizes *downside* volatility (negative-return days), since upside swings aren't typically considered "risk". |
| **Maximum Drawdown** | The single largest peak-to-trough decline in portfolio value over the period. |
| **Calmar Ratio** | CAGR divided by the magnitude of the maximum drawdown -- return relative to the worst historical loss. |
| **Best / Worst Year** | The best and worst calendar-year total returns within the backtest window. |
                """
            )

        with tab_ai:
            st.markdown(
                """
The 💬 button on the right edge of the screen opens **Ask AI About This Portfolio**,
which sends your allocation, backtest period, and computed metrics to a local
**Llama 3.1 8B** model running through [Ollama](https://ollama.com), and asks it to
explain the results in plain language.

This runs entirely on your own machine -- no data leaves your computer. The model is
instructed to explain concepts and trade-offs only, and explicitly told **not** to give
financial advice, recommend trades, or predict future returns.

**Requirements to use this feature:**
- [Ollama](https://ollama.com/download) installed and running locally (`ollama serve`)
- The model pulled once via `ollama pull llama3.1:8b`
- You'll need to run a backtest first, so there's something for the AI to analyze.
                """
            )

        st.info(
            "⚠️ Past performance does not guarantee future results. All figures are "
            "historical and purely illustrative."
        )


# --------------------------------------------------------------------------
# AI portfolio analysis: floating button + popup dialog
# --------------------------------------------------------------------------
def build_ai_summary(
    weights: dict,
    stats: dict,
    window: dict,
    rebalance_frequency: str,
    initial_investment: float,
    final_value: float,
) -> dict:
    """
    Build the plain-language, JSON-safe payload sent to the local AI model.

    Every value is pre-formatted as a display string (e.g. "10.4%", "0.82")
    rather than a raw numpy/NaN value -- this keeps the prompt readable for
    the model and avoids passing non-standard JSON tokens like `NaN`.
    """
    return {
        "allocation": {
            "stocks": f"{weights['Stocks'] * 100:.0f}%",
            "bonds": f"{weights['Bonds'] * 100:.0f}%",
            "gold": f"{weights['Gold'] * 100:.0f}%",
        },
        "backtest_period": {
            "start": window["start"].strftime("%Y-%m-%d"),
            "end": window["end"].strftime("%Y-%m-%d"),
            "years": f"{window['years']:.1f}",
        },
        "rebalancing_frequency": rebalance_frequency,
        "initial_investment": f"${initial_investment:,.0f}",
        "final_value": f"${final_value:,.0f}",
        "metrics": {
            "CAGR": fmt_pct(stats["cagr"]),
            "Annualized Volatility": fmt_pct(stats["volatility"]),
            "Sharpe Ratio": fmt_ratio(stats["sharpe"]),
            "Sortino Ratio": fmt_ratio(stats["sortino"]),
            "Maximum Drawdown": fmt_pct(stats["max_drawdown"]),
            "Calmar Ratio": fmt_ratio(stats["calmar"]),
            "Best Year": fmt_pct(stats["best_year"]),
            "Worst Year": fmt_pct(stats["worst_year"]),
        },
    }


@st.dialog("💬 Ask AI About This Portfolio", width="large")
def ai_chat_dialog(portfolio_summary: dict | None) -> None:
    """
    Popup dialog (triggered by the floating chat button) containing the
    AI question form. Connection/runtime errors from Ollama (e.g. the
    server isn't running) are caught and shown as a friendly message
    instead of crashing the app.
    """
    if portfolio_summary is None:
        st.info(
            "Run a backtest first (set your allocation to exactly 100% in the "
            "sidebar, then click **Run Backtest**) so there's something for the "
            "AI to analyze."
        )
        return

    st.caption(
        "Runs locally via Ollama (Llama 3.1 8B). See the 'AI Analysis' tab in the "
        "Information section for details and requirements."
    )

    question = st.text_input(
        "Ask a question:",
        placeholder=f"Example: {DEFAULT_QUESTION}",
    )

    if st.button("Analyze Portfolio", type="primary"):
        effective_question = question.strip() or DEFAULT_QUESTION
        try:
            with st.spinner("Llama is analyzing..."):
                answer = analyze_portfolio(portfolio_summary, effective_question)
            st.markdown(answer)
        except ConnectionError:
            st.error(
                "Couldn't reach Ollama. Make sure it's installed and running locally "
                "(`ollama serve`) and that the `llama3.1:8b` model has been pulled "
                "(`ollama pull llama3.1:8b`), then try again."
            )
        except Exception as e:  # noqa: BLE001 - surface unexpected errors to the user
            st.error(f"AI analysis failed: {e}")


def render_ai_fab(portfolio_summary: dict | None) -> None:
    """
    Render the floating chat-bubble button pinned to the middle-right edge
    of the screen. Clicking it opens the AI dialog. Disabled (but still
    visible) until a backtest has been run.
    """
    st.markdown(AI_FAB_CSS, unsafe_allow_html=True)
    clicked = st.button(
        "💬",
        key="ai_fab_button",
        help="Ask AI about this portfolio" if portfolio_summary else "Run a backtest first",
    )
    if clicked:
        ai_chat_dialog(portfolio_summary)


# --------------------------------------------------------------------------
# Main app
# --------------------------------------------------------------------------
def main() -> None:
    st.title("📊 Portfolio Allocation Explorer")
    st.caption(
        "An educational tool for exploring how allocation between stocks, bonds, "
        "and gold has historically affected portfolio outcomes."
    )

    render_information_section()

    # Fetch (and clean/cache) price data up front so it's ready the moment a
    # backtest is requested. The "Backtesting Window" section itself is
    # rendered later, at the very end of the results flow.
    try:
        with st.spinner("Loading historical price data..."):
            prices = get_asset_prices()
            window = get_backtest_window(prices)
    except ValueError as e:
        st.error(f"Could not load price data: {e}")
        return
    except Exception as e:  # noqa: BLE001 - surface unexpected errors to the user
        st.error(f"An unexpected error occurred while loading price data: {e}")
        return

    inputs = render_sidebar()

    # Keep the most recent successful backtest in session state so the
    # dashboard persists across reruns that aren't triggered by the button
    # (e.g. widget interactions elsewhere on the page).
    if inputs["run_clicked"]:
        errors = validate_inputs(inputs)
        if errors:
            for err in errors:
                st.error(err)
        else:
            try:
                with st.spinner("Running backtest..."):
                    portfolio = run_backtest(
                        prices=prices,
                        weights=inputs["weights"],
                        initial_investment=inputs["initial_investment"],
                        rebalance_frequency=inputs["rebalance_frequency"],
                    )
                st.session_state["portfolio"] = portfolio
                st.session_state["weights"] = inputs["weights"]
                st.session_state["rebalance_frequency"] = inputs["rebalance_frequency"]
                st.session_state["initial_investment"] = inputs["initial_investment"]
            except ValueError as e:
                st.error(f"Could not run backtest: {e}")
            except Exception as e:  # noqa: BLE001 - surface unexpected errors to the user
                st.error(f"An unexpected error occurred: {e}")

    if "portfolio" not in st.session_state:
        render_ai_fab(None)
        st.info(
            "👈 Set your allocation (must total exactly 100%) and backtest "
            "settings in the sidebar -- or use a quick preset -- then click "
            "**Run Backtest** to begin."
        )
        st.caption(
            f"Data available: {window['start'].strftime('%b %d, %Y')} – "
            f"{window['end'].strftime('%b %d, %Y')} (~{window['years']:.1f} years)."
        )
        return

    portfolio = st.session_state["portfolio"]
    weights = st.session_state["weights"]
    rebalance_frequency = st.session_state["rebalance_frequency"]
    initial_investment = st.session_state["initial_investment"]

    stats = summarize(portfolio)
    final_value = portfolio["Total"].iloc[-1]

    portfolio_summary = build_ai_summary(
        weights, stats, window, rebalance_frequency, initial_investment, final_value
    )
    render_ai_fab(portfolio_summary)

    # --- 1. Backtest results graph, front and center -------------------
    st.header("Backtest Results")
    summary_cols = st.columns([2, 1, 1])
    summary_cols[0].markdown(
        f"### {weights['Stocks']*100:.0f}% Stocks / "
        f"{weights['Bonds']*100:.0f}% Bonds / {weights['Gold']*100:.0f}% Gold"
    )
    summary_cols[0].caption(f"Rebalancing: {rebalance_frequency}")
    summary_cols[1].metric("Initial Investment", f"${initial_investment:,.0f}")
    summary_cols[2].metric(
        "Final Value",
        f"${final_value:,.0f}",
        delta=f"{(final_value / initial_investment - 1) * 100:.1f}%",
    )
    st.plotly_chart(build_growth_chart(portfolio), width="stretch")

    # --- 2. Metrics and other charts ------------------------------------
    st.divider()
    st.subheader("Performance Metrics")
    render_metric_cards(stats)

    st.divider()
    st.plotly_chart(build_drawdown_chart(portfolio), width="stretch")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Annual Returns")
        st.dataframe(
            build_annual_returns_table(portfolio),
            width="stretch",
            hide_index=True,
        )
    with right:
        st.plotly_chart(build_allocation_pie(weights), width="stretch")

    # --- 3. Backtesting timeframe, at the end ---------------------------
    st.divider()
    render_backtest_window(window)

    st.divider()
    st.caption(
        "⚠️ **Disclaimer:** This tool is for educational purposes only and does not "
        "constitute financial advice. All results are based on historical data and "
        "past performance does not guarantee future results. Consult a licensed "
        "financial advisor before making investment decisions."
    )


if __name__ == "__main__":
    main()
