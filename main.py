
import os
import time
import json
import threading
import requests
import pandas as pd
import numpy as np
import websocket

from flask import Flask

# ==========================================
# 1. CONFIGURATION
# ==========================================

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("CHAT_ID", "")

SYMBOLS = [
    "btcusdt",
    "ethusdt",
    "solusdt",
    "bnbusdt",
    "xrpusdt"
]

INTERVAL = "15m"
HISTORY_LIMIT = 250
COOLDOWN_SECONDS = 1800

API_URL = "https://data-api.binance.vision/api/v3/klines"

WS_URL = (
    "wss://data-stream.binance.vision/stream?streams="
)

market_data = {}
last_alert_time = {}

data_lock = threading.Lock()

app = Flask(__name__)


# ==========================================
# 2. FLASK WEB SERVER
# ==========================================

@app.route("/")
def home():
    return "Crypto Trading Signal Bot is Running!"


@app.route("/health")
def health():
    with data_lock:
        loaded = list(market_data.keys())

    return {
        "status": "running",
        "symbols": SYMBOLS,
        "loaded_symbols": loaded,
        "interval": INTERVAL
    }


def run_web_server():

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port,
        use_reloader=False
    )


# ==========================================
# 3. TELEGRAM MESSAGE
# ==========================================

def send_telegram_message(message):

    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("Telegram credentials missing!")
        return False

    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }

    try:

        response = requests.post(
            url,
            json=payload,
            timeout=15
        )

        response.raise_for_status()

        result = response.json()

        if result.get("ok"):

            print("Telegram message sent!")
            return True

        print("Telegram API Error:", result)

        return False

    except Exception as e:

        print("Telegram Error:", e)

        return False


# ==========================================
# 4. LOAD HISTORICAL DATA
# ==========================================

def load_initial_history():

    for sym in SYMBOLS:

        params = {
            "symbol": sym.upper(),
            "interval": INTERVAL,
            "limit": HISTORY_LIMIT
        }

        try:

            response = requests.get(
                API_URL,
                params=params,
                timeout=20
            )

            response.raise_for_status()

            raw_data = response.json()

            if not isinstance(raw_data, list) or not raw_data:
                print(f"Invalid API response: {sym}")
                continue

            df = pd.DataFrame(
                raw_data,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "qav",
                    "trades",
                    "tbb",
                    "tbq",
                    "ignore"
                ]
            )

            for col in [
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]:

                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

            df["timestamp"] = pd.to_numeric(
                df["timestamp"],
                errors="coerce"
            )

            df = df.dropna()

            df = df[
                [
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume"
                ]
            ].reset_index(drop=True)

            if len(df) < 205:
                print(f"Insufficient history: {sym}")
                continue

            with data_lock:
                market_data[sym] = df

            print(
                f"Loaded {len(df)} candles for {sym.upper()}"
            )

        except Exception as e:

            print(f"History loading error {sym}: {e}")


# ==========================================
# 5. INDICATOR CALCULATIONS
# ==========================================

def calculate_indicators(df):

    close = df["close"]

    # EMA 50
    ema50 = close.ewm(
        span=50,
        adjust=False
    ).mean().iloc[-1]

    # EMA 200
    ema200 = close.ewm(
        span=200,
        adjust=False
    ).mean().iloc[-1]

    # RSI 14 - Wilder Smoothing
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    current_gain = avg_gain.iloc[-1]
    current_loss = avg_loss.iloc[-1]

    if current_loss == 0:

        rsi = 100 if current_gain > 0 else 50

    else:

        rs = current_gain / current_loss

        rsi = 100 - (100 / (1 + rs))

    # ATR 14
    high_low = df["high"] - df["low"]

    high_close = (
        df["high"] - df["close"].shift()
    ).abs()

    low_close = (
        df["low"] - df["close"].shift()
    ).abs()

    true_range = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    ).max(axis=1)

    atr = true_range.rolling(
        window=14
    ).mean().iloc[-1]

    # Volume Spike
    avg_volume = df["volume"].rolling(
        window=20
    ).mean().iloc[-2]

    current_volume = df["volume"].iloc[-1]

    return {
        "ema50": ema50,
        "ema200": ema200,
        "rsi": rsi,
        "atr": atr,
        "avg_volume": avg_volume,
        "current_volume": current_volume
    }


# ==========================================
# 6. SIGNAL DETECTION
# ==========================================

def check_signal(sym):

    current_time = time.time()

    # Cooldown
    if sym in last_alert_time:

        elapsed = (
            current_time - last_alert_time[sym]
        )

        if elapsed < COOLDOWN_SECONDS:
            return

    with data_lock:

        df = market_data.get(sym)

        if df is None or len(df) < 205:
            return

        df = df.copy()

    try:

        indicators = calculate_indicators(df)

        ema50 = indicators["ema50"]
        ema200 = indicators["ema200"]

        rsi = indicators["rsi"]
        atr = indicators["atr"]

        avg_volume = indicators["avg_volume"]

        current_volume = indicators["current_volume"]

        price = df["close"].iloc[-1]

        values = [
            ema50,
            ema200,
            rsi,
            atr,
            avg_volume,
            current_volume,
            price
        ]

        if not all(np.isfinite(x) for x in values):
            return

        if atr <= 0 or avg_volume <= 0:
            return

        volume_ratio = current_volume / avg_volume

        is_volume_spike = volume_ratio > 1.2

        symbol = sym.upper()

        # ==================================
        # BUY / LONG CONDITIONS
        # ==================================

        if (
            price > ema200
            and rsi < 32
            and price >= ema50 * 0.992
            and is_volume_spike
        ):

            sl = price - (1.5 * atr)

            tp = price + (3.0 * atr)

            risk_pct = (
                (price - sl) / price
            ) * 100

            reward_pct = (
                (tp - price) / price
            ) * 100

            msg = (
                f"🟢 *STRONG BUY / LONG: {symbol}*\n\n"
                f"💰 *Entry:* `{price:.6f}`\n"
                f"🛑 *Stop Loss:* `{sl:.6f}`\n"
                f"🎯 *Take Profit:* `{tp:.6f}`\n\n"
                f"📊 *RSI (14):* {rsi:.2f}\n"
                f"📈 *Trend:* Above 200 EMA\n"
                f"🔥 *Volume Spike:* {volume_ratio:.2f}x\n"
                f"⚖️ *Risk/Reward:* 1:2\n\n"
                f"📉 *Risk:* {risk_pct:.2f}%\n"
                f"📈 *Potential Reward:* {reward_pct:.2f}%\n"
                f"⏱ *Timeframe:* {INTERVAL}"
            )

            if send_telegram_message(msg):

                last_alert_time[sym] = current_time

                print(f"BUY SIGNAL: {symbol}")

        # ==================================
        # SELL / SHORT CONDITIONS
        # ==================================

        elif (
            price < ema200
            and rsi > 68
            and price <= ema50 * 1.008
            and is_volume_spike
        ):

            sl = price + (1.5 * atr)

            tp = price - (3.0 * atr)

            risk_pct = (
                (sl - price) / price
            ) * 100

            reward_pct = (
                (price - tp) / price
            ) * 100

            msg = (
                f"🔴 *STRONG SELL / SHORT: {symbol}*\n\n"
                f"💰 *Entry:* `{price:.6f}`\n"
                f"🛑 *Stop Loss:* `{sl:.6f}`\n"
                f"🎯 *Take Profit:* `{tp:.6f}`\n\n"
                f"📊 *RSI (14):* {rsi:.2f}\n"
                f"📉 *Trend:* Below 200 EMA\n"
                f"🔥 *Volume Spike:* {volume_ratio:.2f}x\n"
                f"⚖️ *Risk/Reward:* 1:2\n\n"
                f"📈 *Risk:* {risk_pct:.2f}%\n"
                f"📉 *Potential Reward:* {reward_pct:.2f}%\n"
                f"⏱ *Timeframe:* {INTERVAL}"
            )

            if send_telegram_message(msg):

                last_alert_time[sym] = current_time

                print(f"SELL SIGNAL: {symbol}")

    except Exception as e:

        print(
            f"Signal calculation error {sym}: {e}"
        )


# ==========================================
# 7. WEBSOCKET MESSAGE HANDLER
# ==========================================

def on_message(ws, message):

    try:

        payload = json.loads(message)

        data = payload.get("data", payload)

        if "k" not in data:
            return

        kline = data["k"]

        sym = data["s"].lower()

        if sym not in market_data:
            return

        ts = int(kline["t"])

        o = float(kline["o"])
        h = float(kline["h"])
        l = float(kline["l"])
        c = float(kline["c"])
        v = float(kline["v"])

        candle_closed = kline["x"]

        with data_lock:

            df = market_data[sym]

            if int(df["timestamp"].iloc[-1]) == ts:

                df.loc[
                    df.index[-1],
                    [
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume"
                    ]
                ] = [
                    ts, o, h, l, c, v
                ]

            elif ts > int(df["timestamp"].iloc[-1]):

                new_row = pd.DataFrame(
                    [[ts, o, h, l, c, v]],
                    columns=[
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume"
                    ]
                )

                df = pd.concat(
                    [df, new_row],
                    ignore_index=True
                )

                df = df.tail(
                    HISTORY_LIMIT
                ).reset_index(drop=True)

                market_data[sym] = df

            else:
                return

        print(
            f"{sym.upper()} | "
            f"Price: {c} | "
            f"Closed: {candle_closed}"
        )

        # Signal only after candle closes
        if candle_closed:

            check_signal(sym)

    except Exception as e:

        print("WebSocket message error:", e)


# ==========================================
# 8. WEBSOCKET EVENTS
# ==========================================

def on_error(ws, error):

    print("WebSocket Error:", error)


def on_close(ws, close_status_code, close_msg):

    print(
        "WebSocket Closed:",
        close_status_code,
        close_msg
    )


def on_open(ws):

    print("Binance WebSocket Connected!")

    send_telegram_message(
        "✅ *Binance WebSocket Connected!*\n"
        "Market monitoring has started."
    )


# ==========================================
# 9. START WEBSOCKET
# ==========================================

def start_websocket():

    # Retry until all symbols have history
    while True:

        load_initial_history()

        with data_lock:
            loaded_count = len(market_data)

        if loaded_count == len(SYMBOLS):
            break

        print(
            f"History incomplete: "
            f"{loaded_count}/{len(SYMBOLS)} loaded. "
            f"Retrying in 10 seconds..."
        )

        time.sleep(10)

    send_telegram_message(
        "🚀 *Crypto Trading Bot Started!*\n\n"
        "✅ 5 Coins Monitoring\n"
        "✅ 15 Minute Timeframe\n"
        "✅ EMA 50 / EMA 200\n"
        "✅ RSI 14\n"
        "✅ Volume Spike Detection\n"
        "✅ ATR Based SL/TP\n"
        "✅ Automatic WebSocket Reconnection\n\n"
        "Bot is now running!"
    )

    streams = "/".join(
        [
            f"{s}@kline_{INTERVAL}"
            for s in SYMBOLS
        ]
    )

    ws_url = WS_URL + streams

    while True:

        try:

            ws = websocket.WebSocketApp(
                ws_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )

            ws.run_forever(
                ping_interval=20,
                ping_timeout=10
            )

        except Exception as e:

            print(
                "WebSocket reconnecting:",
                e
            )

        print("Reconnecting in 5 seconds...")

        time.sleep(5)


# ==========================================
# 10. MAIN PROGRAM
# ==========================================

if __name__ == "__main__":

    if not TELEGRAM_TOKEN or not CHAT_ID:

        print(
            "WARNING: Telegram credentials are missing!"
        )

    threading.Thread(
        target=start_websocket,
        daemon=True
    ).start()

    run_web_server()
