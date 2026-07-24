import streamlit as st

from backtest_engine import (
    PAIR_TO_OANDA,
    _format_trade_log_dataframe,
    _style_trade_log,
    compute_performance_stats,
    load_market_data,
    parse_reward_ratio,
    run_backtest,
)

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="Tradepulse V", layout="wide")

# -----------------------------
# DARK THEME + PREMIUM STYLING
# -----------------------------
st.markdown(
    """
    <style>
        .stApp {
            background-color: #05070b;
            color: #e2e8f0;
        }

        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 1.5rem;
            max-width: 1500px;
        }

        .control-panel {
            background: #ffffff;
            border: 1px solid #e6eaf2;
            border-radius: 18px;
            padding: 1rem;
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.18);
        }

        .control-title {
            color: #0f172a;
            font-size: 0.96rem;
            font-weight: 900;
            letter-spacing: 0.09em;
            margin-bottom: 0.75rem;
        }

        .results-shell {
            background: #070b12;
            border: 1px solid #182235;
            border-radius: 18px;
            padding: 1rem;
            box-shadow: 0 14px 34px rgba(0, 0, 0, 0.35);
        }

        .results-title {
            color: #e2e8f0;
            font-size: 1.0rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            margin-bottom: 0.85rem;
        }

        .helper-text {
            color: #94a3b8;
            font-size: 0.95rem;
            margin-top: 0.35rem;
        }

        .strategy-card {
            background: #0c1422;
            border: 1px solid #1e2a40;
            border-radius: 14px;
            padding: 0.85rem 0.95rem;
            margin-top: 0.35rem;
            margin-bottom: 0.65rem;
        }

        .strategy-card .name {
            color: #f8fafc;
            font-size: 0.96rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }

        .strategy-card .desc {
            color: #cbd5e1;
            font-size: 0.88rem;
            line-height: 1.35;
            margin-bottom: 0.18rem;
        }

        .strategy-card .cond {
            color: #93a4bf;
            font-size: 0.82rem;
        }

        div[data-testid="stMetric"] {
            background: #0c1422;
            border: 1px solid #1e2a40;
            border-radius: 14px;
            padding: 0.8rem 0.9rem;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.01);
        }

        div[data-testid="stMetricLabel"] {
            color: #93a4bf;
            font-weight: 600;
        }

        div[data-testid="stMetricValue"] {
            color: #f8fafc;
            font-weight: 800;
        }

        .stButton > button {
            width: 100%;
            border: none;
            border-radius: 12px;
            font-weight: 800;
            font-size: 1rem;
            padding: 0.78rem 1rem;
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            color: white;
            box-shadow: 0 10px 24px rgba(37, 99, 235, 0.35);
        }

        .stButton > button:hover {
            filter: brightness(1.03);
            transform: translateY(-1px);
        }

        [data-testid="stVerticalBlock"] .control-panel + div label,
        .control-panel label {
            color: #1e293b !important;
            font-weight: 600;
        }

        .control-panel [data-baseweb="select"] > div {
            background: #ffffff;
            color: #0f172a;
            border-color: #d7deea;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

STRATEGY_INFO = {
    "Support & Resistance": {
        "desc": "Looks for breakouts when price moves above resistance or below support.",
        "cond": "Best for markets that are starting to trend after breaking a key level.",
    },
    "Range Trading Strategy": {
        "desc": "Looks for opportunities when price moves between support and resistance levels. Buys near support areas and sells near resistance areas.",
        "cond": "Best for sideways or range-bound markets.",
    },
    "Pullback Trading Strategy": {
        "desc": "Looks for entries after price temporarily moves against the main trend before continuing in the original direction.",
        "cond": "Best for trending markets with short-term retracements.",
    },
    "Breakout Trading Strategy": {
        "desc": "Looks for opportunities when price breaks above resistance or below support, signaling a possible new market move.",
        "cond": "Best for strong expansion moves after consolidation.",
    },
    "Trend Following Strategy": {
        "desc": "Looks for opportunities by trading in the direction of the overall market trend once upward or downward continuation is confirmed.",
        "cond": "Best for clearly trending markets with sustained direction.",
    },
}

for key, default in [
    ("backtest_ran", False),
    ("results", {}),
    ("equity", None),
    ("data_source", ""),
    ("trade_log", []),
    ("selected_risk", 0.0),
    ("selected_reward", "2:1"),
    ("selected_balance", 0.0),
    ("volume_optimization_suggestion", None),
    ("use_optimized_return", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

left_col, center_col = st.columns([1.05, 3.2], gap="large")

with left_col:
    st.markdown("<div class='control-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='control-title'>CONTROL</div>", unsafe_allow_html=True)

    selected_balance = st.selectbox(
        "Account Balance",
        options=[1000, 5000, 10000, 25000, 50000, 100000],
        index=2,
        format_func=lambda x: f"${x:,.0f}",
    )
    selected_pair = st.selectbox("Pair", options=list(PAIR_TO_OANDA.keys()), index=0)
    selected_risk = st.selectbox(
        "Risk (%)",
        options=[0.5, 1.0, 2.0, 3.0, 5.0],
        index=1,
        format_func=lambda x: f"{x:g}%",
    )
    selected_reward = st.selectbox(
        "Reward (Risk-to-Reward)",
        options=["1:1", "1.5:1", "2:1", "3:1", "4:1"],
        index=2,
    )
    selected_timeframe = st.selectbox("Timeframe", options=["15M", "30M", "1H"], index=0)

    strategy_options = [
        "Support & Resistance",
        "Range Trading Strategy",
        "Pullback Trading Strategy",
        "Breakout Trading Strategy",
        "Trend Following Strategy",
    ]
    selected_strategy = st.selectbox("Strategy", options=strategy_options, index=0)

    st.markdown("<div style='height: 0.2rem;'></div>", unsafe_allow_html=True)
    launch = st.button("🚀 Launch Backtest", use_container_width=True)

    st.markdown(
        f"""
        <div class='strategy-card'>
            <div class='name'>{selected_strategy}</div>
            <div class='desc'>{STRATEGY_INFO[selected_strategy]['desc']}</div>
            <div class='cond'>Market condition: {STRATEGY_INFO[selected_strategy]['cond']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

if launch:
    try:
        candles, source = load_market_data(selected_pair, selected_timeframe)
        reward_ratio = parse_reward_ratio(selected_reward)
        trades, equity_df, ending_balance = run_backtest(
            candles=candles,
            initial_balance=float(selected_balance),
            risk_pct=float(selected_risk),
            reward_ratio=reward_ratio,
            strategy_name=selected_strategy,
        )

        stats = compute_performance_stats(
            trades=trades,
            equity=equity_df,
            initial_balance=float(selected_balance),
            ending_balance=float(ending_balance),
        )
        st.session_state.backtest_ran = True
        st.session_state.results = stats
        st.session_state.equity = equity_df
        st.session_state.data_source = source
        st.session_state.trade_log = trades
        st.session_state.selected_risk = float(selected_risk)
        st.session_state.selected_reward = selected_reward
        st.session_state.selected_balance = float(selected_balance)
    except Exception as e:
        st.session_state.backtest_ran = False
        st.error(f"Failed to load OANDA market data: {e}")

with center_col:
    st.markdown("<div class='results-shell'>", unsafe_allow_html=True)
    st.markdown("<div class='results-title'>BACKTEST RESULTS</div>", unsafe_allow_html=True)

    r = st.session_state.get("results", None)

    if st.session_state.get("backtest_ran", False) and r is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Win Rate", r.get("Win Rate", "N/A"))

        net_profit_display = r.get("Net Profit", "N/A")
        suggestion = st.session_state.get("volume_optimization_suggestion", None)
        if st.session_state.get("use_optimized_return", False) and suggestion is not None:
            net_profit_display = f"{suggestion['optimized_return_percent']:.2f}%"

        c2.metric("Net Profit", net_profit_display)
        c3.metric("Net Return (%)", r.get("Net Return (%)", "N/A"))
        c4.metric("Total Trades", r.get("Total Trades", "N/A"))

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Winning Trades", r.get("Winning Trades", "N/A"))
        c6.metric("Losing Trades", r.get("Losing Trades", "N/A"))
        c7.metric("Profit Factor", r.get("Profit Factor", "N/A"))
        c8.metric("Maximum Drawdown", r.get("Maximum Drawdown", "N/A"))

        c9, c10, c11, c12 = st.columns(4)
        c9.metric("Average Win", r.get("Average Win", "N/A"))
        c10.metric("Average Loss", r.get("Average Loss", "N/A"))
        c11.metric("Largest Win", r.get("Largest Win", "N/A"))
        c12.metric("Largest Loss", r.get("Largest Loss", "N/A"))

        c13, _, _, _ = st.columns(4)
        c13.metric("Ending Account Balance", r.get("Ending Account Balance", "N/A"))

        st.markdown("<div style='height: 0.85rem;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='results-title' style='font-size:0.95rem;'>EQUITY CURVE</div>", unsafe_allow_html=True)
        st.line_chart(st.session_state.equity, use_container_width=True)

        st.markdown("<div style='height: 1.0rem;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='results-title' style='font-size:0.95rem;'>Trade Log</div>", unsafe_allow_html=True)

        trades_for_log = st.session_state.trade_log
        if not trades_for_log:
            st.markdown(
                "<p class='helper-text'>No trades available. Run a backtest to generate a Trade Log.</p>",
                unsafe_allow_html=True,
            )
        else:
            styled_log = _style_trade_log(
                _format_trade_log_dataframe(
                    trades=trades_for_log,
                    risk_pct=st.session_state.selected_risk,
                    reward_label=st.session_state.selected_reward,
                    initial_balance=st.session_state.selected_balance,
                )
            )
            st.dataframe(styled_log, use_container_width=True, height=420)
    else:
        st.warning("No backtest results available. Check your OANDA data connection.")

    st.markdown("</div>", unsafe_allow_html=True)
