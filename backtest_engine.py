import numpy as np
import pandas as pd
import requests
import streamlit as st
from dataclasses import dataclass
from typing import Dict, List, Tuple

PAIR_TO_OANDA = {
    "EUR/USD": "EUR_USD",
    "GBP/USD": "GBP_USD",
    "USD/JPY": "USD_JPY",
    "AUD/USD": "AUD_USD",
    "XAU/USD": "XAU_USD",
    "USD/CAD": "USD_CAD",
}

TIMEFRAME_TO_OANDA = {
    "15M": "M15",
    "30M": "M30",
    "1H": "H1",
}


def _normalize_ohlcv_schema(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        lc = col.lower().strip()
        if lc in ("time", "date", "datetime", "timestamp"):
            rename_map[col] = "timestamp"
        elif lc == "open":
            rename_map[col] = "open"
        elif lc == "high":
            rename_map[col] = "high"
        elif lc == "low":
            rename_map[col] = "low"
        elif lc in ("close", "adj_close", "adj close"):
            rename_map[col] = "close"
        elif lc in ("volume", "vol"):
            rename_map[col] = "volume"

    out = df.rename(columns=rename_map).copy()
    required = ["timestamp", "open", "high", "low", "close", "volume"]

    for c in required:
        if c not in out.columns:
            if c == "volume":
                out[c] = 0.0
            else:
                raise ValueError(f"Missing required OHLCV column: {c}")

    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"])
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out[["timestamp", "open", "high", "low", "close", "volume"]]


def load_oanda_data(pair: str, timeframe: str, count: int = 5000) -> Tuple[pd.DataFrame, str]:
    token = st.secrets.get("OANDA_API_TOKEN", None)
    if not token:
        raise RuntimeError("OANDA_API_TOKEN not found in Streamlit secrets")

    instrument = PAIR_TO_OANDA.get(pair)
    granularity = TIMEFRAME_TO_OANDA.get(timeframe)
    if not instrument or not granularity:
        raise RuntimeError(f"Unsupported pair/timeframe for OANDA: {pair} {timeframe}")

    base_url = st.secrets.get("OANDA_API_URL", "https://api-fxpractice.oanda.com")
    endpoint = f"{base_url}/v3/instruments/{instrument}/candles"

    headers = {"Authorization": f"******"}
    params = {
        "price": "M",
        "granularity": granularity,
        "count": count,
    }

    resp = requests.get(endpoint, headers=headers, params=params, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"OANDA request failed: {resp.status_code} {resp.text}")

    candles = resp.json().get("candles", [])
    if not candles:
        raise RuntimeError("OANDA returned no candles")

    rows = []
    for c in candles:
        if not c.get("complete", False):
            continue
        mid = c.get("mid", {})
        rows.append(
            {
                "timestamp": c.get("time"),
                "open": float(mid.get("o")),
                "high": float(mid.get("h")),
                "low": float(mid.get("l")),
                "close": float(mid.get("c")),
                "volume": float(c.get("volume", 0)),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No complete candles returned from OANDA for {instrument}")

    df = _normalize_ohlcv_schema(df)
    return df, f"oanda:{instrument}:{granularity}"


@st.cache_data(show_spinner=False)
def load_market_data(pair: str, timeframe: str) -> Tuple[pd.DataFrame, str]:
    return load_oanda_data(pair, timeframe)


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_amount: float
    exit_price: float
    pnl: float
    result: str


def parse_reward_ratio(rr: str) -> float:
    left, right = rr.split(":")
    return float(left) / float(right)


def identify_support_resistance(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    out = df.copy()
    out["support"] = out["low"].rolling(lookback).min().shift(1)
    out["resistance"] = out["high"].rolling(lookback).max().shift(1)
    return out


def generate_sr_signal(df: pd.DataFrame, i: int) -> str:
    row = df.iloc[i]
    if pd.isna(row["support"]) or pd.isna(row["resistance"]):
        return "flat"
    if row["close"] > row["resistance"]:
        return "long"
    if row["close"] < row["support"]:
        return "short"
    return "flat"


def generate_range_trading_signal(df: pd.DataFrame, i: int) -> str:
    row = df.iloc[i]
    support = row.get("support")
    resistance = row.get("resistance")

    if pd.isna(support) or pd.isna(resistance):
        return "flat"

    range_width = float(resistance - support)
    if range_width <= 0 or not (support <= row["close"] <= resistance):
        return "flat"

    zone = range_width * 0.25
    near_support = row["close"] <= support + zone
    near_resistance = row["close"] >= resistance - zone
    prev_close = df.iloc[i - 1]["close"] if i > 0 else np.nan

    if near_support and (pd.isna(prev_close) or row["close"] >= prev_close):
        return "long"
    if near_resistance and (pd.isna(prev_close) or row["close"] <= prev_close):
        return "short"
    return "flat"


def generate_pullback_trading_signal(df: pd.DataFrame, i: int) -> str:
    if i < 3:
        return "flat"

    row, prev, prev2 = df.iloc[i], df.iloc[i - 1], df.iloc[i - 2]
    if pd.isna(row["support"]) or pd.isna(row["resistance"]):
        return "flat"

    bullish_trend = (
        row["close"] > prev["close"] > prev2["close"]
        and row["high"] >= prev["high"]
        and prev["high"] >= prev2["high"]
        and row["low"] >= prev["low"]
    )
    bearish_trend = (
        row["close"] < prev["close"] < prev2["close"]
        and row["low"] <= prev["low"]
        and prev["low"] <= prev2["low"]
        and row["high"] <= prev["high"]
    )

    pullback_zone_up = row["close"] <= row["support"] + (row["resistance"] - row["support"]) * 0.35
    pullback_zone_down = row["close"] >= row["resistance"] - (row["resistance"] - row["support"]) * 0.35
    bullish_confirmation = row["close"] > row["open"] and row["close"] > prev["high"]
    bearish_confirmation = row["close"] < row["open"] and row["close"] < prev["low"]

    if bullish_trend and pullback_zone_up and bullish_confirmation:
        return "long"
    if bearish_trend and pullback_zone_down and bearish_confirmation:
        return "short"
    return "flat"


def generate_breakout_trading_signal(df: pd.DataFrame, i: int) -> str:
    row = df.iloc[i]
    if pd.isna(row["support"]) or pd.isna(row["resistance"]):
        return "flat"
    if row["close"] > row["resistance"]:
        return "long"
    if row["close"] < row["support"]:
        return "short"
    return "flat"


def generate_trend_following_signal(df: pd.DataFrame, i: int) -> str:
    if i < 3:
        return "flat"

    row = df.iloc[i]
    prev = df.iloc[i - 1]
    prev2 = df.iloc[i - 2]

    bullish_trend = (
        row["close"] > prev["close"] > prev2["close"]
        and row["high"] > prev["high"]
        and prev["high"] >= prev2["high"]
        and row["low"] > prev["low"]
    )
    bearish_trend = (
        row["close"] < prev["close"] < prev2["close"]
        and row["low"] < prev["low"]
        and prev["low"] <= prev2["low"]
        and row["high"] < prev["high"]
    )

    bullish_confirmation = row["close"] > row["open"] and row["close"] >= prev["close"]
    bearish_confirmation = row["close"] < row["open"] and row["close"] <= prev["close"]

    if bullish_trend and bullish_confirmation:
        return "long"
    if bearish_trend and bearish_confirmation:
        return "short"
    return "flat"


def run_backtest(
    candles: pd.DataFrame,
    initial_balance: float,
    risk_pct: float,
    reward_ratio: float,
    strategy_name: str,
) -> Tuple[List[Trade], pd.DataFrame, float]:
    df = identify_support_resistance(candles, lookback=20)

    balance = initial_balance
    trades: List[Trade] = []
    equity_points = [{"timestamp": df.iloc[0]["timestamp"], "equity": balance}]

    i, n = 21, len(df)
    while i < n - 1:
        signal = "flat"
        if strategy_name == "Support & Resistance":
            signal = generate_sr_signal(df, i)
        elif strategy_name == "Range Trading Strategy":
            signal = generate_range_trading_signal(df, i)
        elif strategy_name == "Pullback Trading Strategy":
            signal = generate_pullback_trading_signal(df, i)
        elif strategy_name == "Breakout Trading Strategy":
            signal = generate_breakout_trading_signal(df, i)
        elif strategy_name == "Trend Following Strategy":
            signal = generate_trend_following_signal(df, i)

        if signal == "flat":
            i += 1
            continue

        entry_candle = df.iloc[i + 1]
        entry_price = float(entry_candle["open"])

        atr_window = df.iloc[max(0, i - 13): i + 1]
        atr = float((atr_window["high"] - atr_window["low"]).mean())
        if atr <= 0 or np.isnan(atr):
            i += 1
            continue

        stop_distance = max(atr, entry_price * 0.001)
        if signal == "long":
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + stop_distance * reward_ratio
            per_unit_risk = entry_price - stop_loss
        else:
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - stop_distance * reward_ratio
            per_unit_risk = stop_loss - entry_price

        risk_amount = balance * (risk_pct / 100.0)
        if per_unit_risk <= 0:
            i += 1
            continue

        position_size = risk_amount / per_unit_risk
        exit_price = entry_price
        exit_time = entry_candle["timestamp"]
        result = "loss"

        j = i + 1
        while j < n:
            c = df.iloc[j]
            hi = float(c["high"])
            lo = float(c["low"])
            if signal == "long":
                hit_sl = lo <= stop_loss
                hit_tp = hi >= take_profit
                if hit_sl or hit_tp:
                    exit_price = stop_loss if hit_sl else take_profit
                    result = "loss" if hit_sl else "win"
                    exit_time = c["timestamp"]
                    break
            else:
                hit_sl = hi >= stop_loss
                hit_tp = lo <= take_profit
                if hit_sl or hit_tp:
                    exit_price = stop_loss if hit_sl else take_profit
                    result = "loss" if hit_sl else "win"
                    exit_time = c["timestamp"]
                    break
            j += 1

        pnl = (exit_price - entry_price) * position_size if signal == "long" else (entry_price - exit_price) * position_size
        balance += pnl
        trades.append(
            Trade(
                entry_time=entry_candle["timestamp"],
                exit_time=exit_time,
                side=signal,
                entry_price=entry_price,
                stop_loss=float(stop_loss),
                take_profit=float(take_profit),
                position_size=float(position_size),
                risk_amount=float(risk_amount),
                exit_price=float(exit_price),
                pnl=float(pnl),
                result=result,
            )
        )
        equity_points.append({"timestamp": exit_time, "equity": balance})
        i = j + 1

    equity_df = (
        pd.DataFrame(equity_points)
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .set_index("timestamp")
    )
    return trades, equity_df, balance


def compute_performance_stats(trades: List[Trade], equity: pd.DataFrame, initial_balance: float, ending_balance: float) -> Dict[str, str]:
    if len(trades) == 0:
        return {
            "Win Rate": "0.0%",
            "Net Profit": "$0.00",
            "Net Return (%)": "0.00%",
            "Ending Account Balance": f"${initial_balance:,.2f}",
            "Total Trades": "0",
            "Winning Trades": "0",
            "Losing Trades": "0",
            "Profit Factor": "0.00",
            "Maximum Drawdown": "0.00%",
            "Average Win": "$0.00",
            "Average Loss": "$0.00",
            "Largest Win": "$0.00",
            "Largest Loss": "$0.00",
        }

    pnls = np.array([t.pnl for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    total = len(trades)
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / total) * 100 if total else 0.0
    net_profit = ending_balance - initial_balance
    net_return = (net_profit / initial_balance) * 100 if initial_balance else 0.0
    gross_profit = wins.sum() if len(wins) else 0.0
    gross_loss_abs = abs(losses.sum()) if len(losses) else 0.0
    profit_factor = (gross_profit / gross_loss_abs) if gross_loss_abs > 0 else np.inf
    curve = equity["equity"].values.astype(float)
    running_peak = np.maximum.accumulate(curve)
    drawdowns = (curve - running_peak) / running_peak
    max_dd = float(drawdowns.min()) if len(drawdowns) else 0.0

    return {
        "Win Rate": f"{win_rate:.1f}%",
        "Net Profit": f"${net_profit:,.2f}",
        "Net Return (%)": f"{net_return:.2f}%",
        "Ending Account Balance": f"${ending_balance:,.2f}",
        "Total Trades": f"{total}",
        "Winning Trades": f"{win_count}",
        "Losing Trades": f"{loss_count}",
        "Profit Factor": "∞" if np.isinf(profit_factor) else f"{profit_factor:.2f}",
        "Maximum Drawdown": f"{max_dd * 100:.2f}%",
        "Average Win": f"${(wins.mean() if len(wins) else 0.0):,.2f}",
        "Average Loss": f"${(losses.mean() if len(losses) else 0.0):,.2f}",
        "Largest Win": f"${(wins.max() if len(wins) else 0.0):,.2f}",
        "Largest Loss": f"${(losses.min() if len(losses) else 0.0):,.2f}",
    }


def _format_trade_log_dataframe(trades: List[Trade], risk_pct: float, reward_label: str, initial_balance: float) -> pd.DataFrame:
    columns = [
        "Trade #",
        "Entry Time",
        "Exit Time",
        "Position (Buy/Sell)",
        "Entry Price",
        "Exit Price",
        "Stop Loss",
        "Take Profit",
        "Risk (%)",
        "Reward (R:R)",
        "Profit/Loss ($)",
        "Profit/Loss (%)",
        "Result (Win/Loss)",
    ]

    if not trades:
        return pd.DataFrame(columns=columns)

    rows = []
    running_balance = float(initial_balance)
    for idx, t in enumerate(trades, start=1):
        pnl_pct = (t.pnl / running_balance * 100.0) if running_balance else 0.0
        rows.append(
            {
                "Trade #": idx,
                "Entry Time": pd.to_datetime(t.entry_time).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "Exit Time": pd.to_datetime(t.exit_time).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "Position (Buy/Sell)": "Buy" if t.side == "long" else "Sell",
                "Entry Price": f"{t.entry_price:.5f}",
                "Exit Price": f"{t.exit_price:.5f}",
                "Stop Loss": f"{t.stop_loss:.5f}",
                "Take Profit": f"{t.take_profit:.5f}",
                "Risk (%)": f"{risk_pct:.2f}%",
                "Reward (R:R)": reward_label,
                "Profit/Loss ($)": f"${t.pnl:,.2f}",
                "Profit/Loss (%)": f"{pnl_pct:+.2f}%",
                "Result (Win/Loss)": "Win" if t.result == "win" else "Loss",
            }
        )
        running_balance += t.pnl

    return pd.DataFrame(rows).sort_values("Trade #", ascending=False).reset_index(drop=True)[columns]


def _style_trade_log(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    def row_style(row):
        result = str(row.get("Result (Win/Loss)", "")).lower()
        bg = "rgba(34, 197, 94, 0.10)" if result == "win" else ("rgba(239, 68, 68, 0.10)" if result == "loss" else "transparent")
        return [f"background-color: {bg};"] * len(row)

    def zebra(col):
        return ["background-color: rgba(148, 163, 184, 0.05);" if i % 2 else "" for i in range(len(col))]

    return (
        df.style
        .apply(zebra, axis=0)
        .apply(row_style, axis=1)
        .set_properties(**{"color": "#e2e8f0", "border-color": "#1e2a40", "font-size": "0.90rem", "text-align": "right", "white-space": "nowrap"})
        .set_properties(subset=["Trade #", "Entry Time", "Exit Time", "Position (Buy/Sell)", "Result (Win/Loss)"], **{"text-align": "left"})
        .set_table_styles(
            [
                {"selector": "thead th", "props": [("background-color", "#0c1422"), ("color", "#cbd5e1"), ("border", "1px solid #1e2a40"), ("font-weight", "700"), ("text-align", "left")]},
                {"selector": "tbody td", "props": [("border", "1px solid #1e2a40")]},
            ]
        )
    )
