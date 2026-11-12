
import os
import time
import json
import threading
import requests
import pandas as pd
import numpy as np
import websocket

from flask import Flask, jsonify


# ==================================================
# 1. CONFIGURATION
# ==================================================

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
LOG_INTERVAL = 30

# Binance Public Market Data REST API
API_URL = "https://data-api.binance.vision/api/v3/klines"

# Binance Public Market Data WebSocket
WS_URL = "wss://data-stream.binance.vision/stream?streams="

# 15 minutes in milliseconds
INTERVAL_MS = 15 * 60 * 1000

# Shared data
market_data = {}
last_alert_time = {}
last_ws_update = {}
last_price_log = {}

data_lock = threading.RLock()

ws_connected = False

app = Flask(__name__)

# Reusable HTTP session
http = requests.Session()

http.headers.update({
    "User-Agent": "CryptoSignalBot/1.0"
})


# ==================================================
# 2. FLASK WEB SERVER
# ==================================================

@app.route("/")
def home():

    return """
    <html>
    <head>
        <title>Crypto Signal Bot</title>
    </head>
    <body>
        <h2>Crypto Trading Signal Bot</h2>
        <p>Status: Running</p>
        <p><a href="/health">Health Check</a></p>
        <p><a href="/prices">Live Prices</a></p>
    </body>
    </html>
    """


@app.route("/health")
def health():

    with data_lock:

        loaded = list(market_data.keys())

        status = {
            "status": "running",
            "websocket_connected": ws_connected,
            "symbols": SYMBOLS,
            "loaded_symbols": loaded,
            "loaded_count": len(loaded),
            "interval": INTERVAL,
            "last_update": {}
        }

        for sym in SYMBOLS:

            update = last_ws_update.get(sym)

            status["last_update"][sym.upper()] = (
                round(time.time() - update, 2)
                if update else None
            )

    return jsonify(status)


@app.route("/prices")
def prices():

    result = {}

    with data_lock:

        for sym, df in market_data.items():

            if df.empty:
                continue

            result[sym.upper()] = {
                "price": float(df["close"].iloc[-1]),
                "volume": float(df["volume"].iloc[-1]),
                "candle_timestamp": int(
                    df["timestamp"].iloc[-1]
                ),
                "last_ws_update_seconds_ago": round(
                    time.time() - last_ws_update.get(
                        sym, time.time()
                    ), 2
                )
            }

    return jsonify({
        "status": "success",
        "websocket_connected": ws_connected,
        "prices": result
    })


def run_web_server():

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True,
        use_reloader=False
    )


# ==================================================
# 3. TELEGRAM NOTIFICATION
# ==================================================

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

        response = http.post(
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

        print("Telegram Error:", repr(e))

        return False


# ==================================================
# 4. LOAD BINANCE HISTORICAL DATA
# ==================================================

def fetch_symbol_history(sym):

    params = {
        "symbol": sym.upper(),
        "interval": INTERVAL,
        "limit": HISTORY_LIMIT
    }

    try:

        response = http.get(
            API_URL,
            params=params,
            timeout=20
        )

        print(
            f"REST API: {sym.upper()} | "
            f"HTTP {response.status_code}"
        )

        if response.status_code != 200:

            print(
                "Binance REST Error:",
                response.text[:500]
            )

            return None

        raw_data = response.json()

        if not isinstance(raw_data, list) or not raw_data:

            print("Invalid Binance response:", sym)

            return None

        columns = [
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

        df = pd.DataFrame(
            raw_data,
            columns=columns
        )

        numeric_columns = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]

        for col in numeric_columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = df.dropna(
            subset=numeric_columns
        )

        df = df[
            numeric_columns
        ].reset_index(drop=True)

        if len(df) < 205:

            print(
                f"Insufficient history: {sym} "
                f"| Candles: {len(df)}"
            )

            return None

        print(
            f"Successfully loaded {len(df)} candles: "
            f"{sym.upper()}"
        )

        return df

    except Exception as e:

        print(
            f"Historical data error {sym}:",
            repr(e)
        )

        return None


def load_initial_history():

    success_count = 0

    for sym in SYMBOLS:

        df = fetch_symbol_history(sym)

        if df is None:
            continue

        with data_lock:

            market_data[sym] = df

        success_count += 1

    print(
        f"Historical data loaded: "
        f"{success_count}/{len(SYMBOLS)}"
    )

    return success_count == len(SYMBOLS)


# ==================================================
# 5. INDICATOR CALCULATIONS
# ==================================================

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

    # RSI 14
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

    # Previous 20-candle average volume
    avg_volume = df["volume"].rolling(
        window=20
    ).mean().iloc[-2]

    # Current candle volume
    current_volume = df["volume"].iloc[-1]

    return {

        "ema50": ema50,
        "ema200": ema200,
        "rsi": rsi,
        "atr": atr,
        "avg_volume": avg_volume,
        "current_volume": current_volume

    }


# ==================================================
# 6. PRICE FORMATTING
# ==================================================

def format_price(value):

    return (
        f"{value:.8f}"
        .rstrip("0")
        .rstrip(".")
    )


# ==================================================
# 7. SIGNAL DETECTION
# ==================================================

def check_signal(sym):

    current_time = time.time()

    # Cooldown check
    previous_alert = last_alert_time.get(sym, 0)

    if current_time - previous_alert < COOLDOWN_SECONDS:
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

        if atr <= 0 or avg_volume <= 0 or price <= 0:
            return

        volume_ratio = current_volume / avg_volume

        is_volume_spike = volume_ratio > 1.2

        symbol = sym.upper()

        entry = format_price(price)

        # ------------------------------------------
        # BUY / LONG
        # ------------------------------------------

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

                f"💰 *Entry:* `{entry}`\n"

                f"🛑 *Stop Loss:* `{format_price(sl)}`\n"

                f"🎯 *Take Profit:* `{format_price(tp)}`\n\n"

                f"📊 *RSI (14):* {rsi:.2f}\n"

                f"📈 *Trend:* Above 200 EMA\n"

                f"🔥 *Volume Spike:* {volume_ratio:.2f}x\n"

                f"⚖️ *Risk/Reward:* 1:2\n\n"

                f"📉 *Risk:* {risk_pct:.2f}%\n"

                f"📈 *Potential Reward:* {reward_pct:.2f}%\n"

                f"⏱ *Timeframe:* {INTERVAL}\n\n"

                f"⚠️ Signal only. Not financial advice."
            )

            if send_telegram_message(msg):

                last_alert_time[sym] = current_time

                print("BUY SIGNAL:", symbol)

        # ------------------------------------------
        # SELL / SHORT
        # ------------------------------------------

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

                f"💰 *Entry:* `{entry}`\n"

                f"🛑 *Stop Loss:* `{format_price(sl)}`\n"

                f"🎯 *Take Profit:* `{format_price(tp)}`\n\n"

                f"📊 *RSI (14):* {rsi:.2f}\n"

                f"📉 *Trend:* Below 200 EMA\n"

                f"🔥 *Volume Spike:* {volume_ratio:.2f}x\n"

                f"⚖️ *Risk/Reward:* 1:2\n\n"

                f"📈 *Risk:* {risk_pct:.2f}%\n"

                f"📉 *Potential Reward:* {reward_pct:.2f}%\n"

                f"⏱ *Timeframe:* {INTERVAL}\n\n"

                f"⚠️ Signal only. Not financial advice."
            )

            if send_telegram_message(msg):

                last_alert_time[sym] = current_time

                print("SELL SIGNAL:", symbol)

    except Exception as e:

        print(
            f"Signal calculation error {sym}:",
            repr(e)
        )


# ==================================================
# 8. UPDATE MARKET DATA
# ==================================================

def update_market_data(
    sym,
    ts,
    o,
    h,
    l,
    c,
    v
):

    need_refresh = False

    with data_lock:

        df = market_data.get(sym)

        if df is None:
            return False

        last_ts = int(
            df["timestamp"].iloc[-1]
        )

        if ts < last_ts:
            return False

        # Detect missing candles after reconnect
        if ts > last_ts + INTERVAL_MS:

            need_refresh = True

    if need_refresh:

        print(
            f"Missing candles detected for {sym.upper()}. "
            "Refreshing REST history..."
        )

        fresh_df = fetch_symbol_history(sym)

        if fresh_df is not None:

            with data_lock:

                market_data[sym] = fresh_df

    # Apply latest WebSocket candle
    with data_lock:

        df = market_data.get(sym)

        if df is None:
            return False

        last_ts = int(
            df["timestamp"].iloc[-1]
        )

        if ts < last_ts:
            return False

        if ts == last_ts:

            idx = df.index[-1]

            df.loc[
                idx,
                [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume"
                ]
            ] = [
                o, h, l, c, v
            ]

        else:

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

            market_data[sym] = df.tail(
                HISTORY_LIMIT
            ).reset_index(drop=True)

    return True


# ==================================================
# 9. WEBSOCKET MESSAGE HANDLER
# ==================================================

def on_message(ws, message):

    try:

        payload = json.loads(message)

        # Combined stream response
        data = payload.get("data", payload)

        if "k" not in data:
            return

        kline = data["k"]

        sym = data.get("s", "").lower()

        if sym not in SYMBOLS:
            return

        ts = int(kline["t"])

        o = float(kline["o"])

        h = float(kline["h"])

        l = float(kline["l"])

        c = float(kline["c"])

        v = float(kline["v"])

        candle_closed = bool(kline["x"])

        updated = update_market_data(
            sym,
            ts,
            o,
            h,
            l,
            c,
            v
        )

        if not updated:
            return

        with data_lock:

            last_ws_update[sym] = time.time()

        # Avoid excessive Render logging
        current_time = time.time()

        previous_log = last_price_log.get(sym, 0)

        if current_time - previous_log >= LOG_INTERVAL:

            print(
                f"{sym.upper()} | "
                f"Live Price: {format_price(c)} | "
                f"Volume: {v:.4f} | "
                f"Closed: {candle_closed}"
            )

            last_price_log[sym] = current_time

        # Signal only after candle closes
        if candle_closed:

            print(
                f"CANDLE CLOSED: {sym.upper()}"
            )

            check_signal(sym)

    except Exception as e:

        print(
            "WebSocket message error:",
            repr(e)
        )


# ==================================================
# 10. WEBSOCKET EVENTS
# ==================================================

def on_open(ws):

    global ws_connected

    with data_lock:

        ws_connected = True

    print("====================================")

    print("BINANCE WEBSOCKET CONNECTED!")

    print("URL:", ws.url)

    print("====================================")

    send_telegram_message(
        "✅ *Binance WebSocket Connected!*\n"
        "Live market monitoring started."
    )


def on_error(ws, error):

    print("BINANCE WEBSOCKET ERROR!")

    print(repr(error))


def on_close(ws, close_status_code, close_msg):

    global ws_connected

    with data_lock:

        ws_connected = False

    print(
        "WebSocket Closed:",
        close_status_code,
        close_msg
    )


# ==================================================
# 11. WEBSOCKET CONNECTION
# ==================================================

def start_websocket():

    # Load history until all symbols are available
    while True:

        success = load_initial_history()

        if success:
            break

        print(
            "Historical data incomplete. "
            "Retrying in 10 seconds..."
        )

        time.sleep(10)

    send_telegram_message(
        "🚀 *Crypto Trading Bot Started!*\n\n"

        "✅ 5 Coins Monitoring\n"

        "✅ 15 Minute Timeframe\n"

        "✅ EMA 50 / EMA 200\n"

        "✅ RSI 14\n"

        "✅ ATR 14\n"

        "✅ Volume Spike Detection\n"

        "✅ ATR Based SL/TP\n"

        "✅ Automatic Reconnection\n\n"

        "Bot is now running!"
    )

    streams = "/".join(
        [
            f"{sym}@kline_{INTERVAL}"
            for sym in SYMBOLS
        ]
    )

    ws_url = WS_URL + streams

    while True:

        try:

            print("Connecting to Binance WebSocket...")

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
                "WebSocket connection exception:",
                repr(e)
            )

        with data_lock:

            global_ws_state = ws_connected

        print(
            "WebSocket disconnected. "
            "Reconnecting in 5 seconds..."
        )

        time.sleep(5)


# ==================================================
# 12. MAIN PROGRAM
# ==================================================

if __name__ == "__main__":

    print("Starting Crypto Trading Signal Bot...")

    if not TELEGRAM_TOKEN or not CHAT_ID:

        print(
            "WARNING: Telegram credentials missing!"
        )

    threading.Thread(
        target=start_websocket,
        daemon=True
    ).start()

    run_web_server()
