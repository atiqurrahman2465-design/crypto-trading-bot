import requests
import pandas as pd
import numpy as np
import time
import threading
import json
import os
import websocket
from flask import Flask

# ==========================================
# ১. আপনার টেলিগ্রাম তথ্য এখানে বসান
# ==========================================
TELEGRAM_TOKEN = 9108825365:AAErPQ1ndyc4ZA2muKTCt2_Sl_vsjh0kLqM
CHAT_ID = 1576421688

# ৫টি কয়েন এবং ১৫ মিনিটের টাইমফ্রেম
SYMBOLS = ["btcusdt", "ethusdt", "solusdt", "bnbusdt", "xrpusdt"]
INTERVAL = "15m"

market_data = {}
last_alert_time = {}
COOLDOWN_SECONDS = 1800  # একই কয়েনে ৩০ মিনিট পরপর অ্যালার্ট দেবে

# Render-কে ২৪/৭ সচল রাখার ওয়েব সার্ভার
app = Flask(__name__)

@app.route('/')
def home():
    return "Crypto 1-Second Live Predictor is Running!"

def run_web_server():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

def send_telegram_message(message):
    url = f"https://telegram.org{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Telegram Error: {e}")

def load_initial_history():
    for sym in SYMBOLS:
        url = f"https://binance.vision{sym.upper()}&interval={INTERVAL}&limit=250"
        try:
            res = requests.get(url, timeout=10).json()
            df = pd.DataFrame(res, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'qav', 'trades', 'tbb', 'tbq', 'ignore'
            ])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            market_data[sym] = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            print(f"History load error for {sym}: {e}")

def check_signal(sym):
    current_time = time.time()
    if sym in last_alert_time and (current_time - last_alert_time[sym] < COOLDOWN_SECONDS):
        return

    df = market_data.get(sym)
    if df is None or len(df) < 205:
        return

    # 50 EMA এবং 200 EMA হিসাব
    ema50 = df['close'].ewm(span=50, adjust=False).mean().iloc[-1]
    ema200 = df['close'].ewm(span=200, adjust=False).mean().iloc[-1]

    # 14 RSI হিসাব
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = (100 - (100 / (1 + rs))).iloc[-1]

    # Volume Spike চেক (গত ২০ ক্যান্ডেলের গড় ভলিউমের ১.২ গুণ)
    avg_volume = df['volume'].rolling(window=20).mean().iloc[-2]
    current_volume = df['volume'].iloc[-1]
    is_volume_spike = current_volume > (avg_volume * 1.2)

    # ATR হিসাব (1:2 Risk-to-Reward নির্ণয়ের জন্য)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = np.max(pd.concat([high_low, high_close, low_close], axis=1), axis=1)
    atr = tr.rolling(14).mean().iloc[-1]

    price = df['close'].iloc[-1]
    symbol_upper = sym.upper()

    # ==========================================
    # BUY / LONG শর্ত (Uptrend + Oversold + Volume)
    # ==========================================
    if (price > ema200) and (rsi < 32) and (price >= ema50 * 0.992) and is_volume_spike:
        sl = price - (1.5 * atr)
        tp = price + (3.0 * atr)  # 1:2 Ratio
        risk_pct = ((price - sl) / price) * 100
        reward_pct = ((tp - price) / price) * 100
        vol_ratio = current_volume / avg_volume

        msg = (
            f"🟢 *STRONG BUY / LONG: {symbol_upper}*\n\n"
            f"💰 *Entry:* `{price:.4f}`\n"
            f"🛑 *Stop-Loss (SL):* `{sl:.4f}` (-{risk_pct:.2f}%)\n"
            f"🎯 *Take-Profit (TP):* `{tp:.4f}` (+{reward_pct:.2f}%)\n\n"
            f"📊 *RSI (14):* {rsi:.2f} | 📈 *Trend:* Above 200 EMA\n"
            f"🔥 *Volume Spike:* {vol_ratio:.1f}x | ⚖️ *R/R:* 1:2"
        )
        send_telegram_message(msg)
        last_alert_time[sym] = current_time

    # ==========================================
    # SELL / SHORT শর্ত (Downtrend + Overbought + Volume)
    # ==========================================
    elif (price < ema200) and (rsi > 68) and (price <= ema50 * 1.008) and is_volume_spike:
        sl = price + (1.5 * atr)
        tp = price - (3.0 * atr)  # 1:2 Ratio
        risk_pct = ((sl - price) / price) * 100
        reward_pct = ((price - tp) / price) * 100
        vol_ratio = current_volume / avg_volume

        msg = (
            f"🔴 *STRONG SELL / SHORT: {symbol_upper}*\n\n"
            f"💰 *Entry:* `{price:.4f}`\n"
            f"🛑 *Stop-Loss (SL):* `{sl:.4f}` (-{risk_pct:.2f}%)\n"
            f"🎯 *Take-Profit (TP):* `{tp:.4f}` (+{reward_pct:.2f}%)\n\n"
            f"📊 *RSI (14):* {rsi:.2f} | 📉 *Trend:* Below 200 EMA\n"
            f"🔥 *Volume Spike:* {vol_ratio:.1f}x | ⚖️ *R/R:* 1:2"
        )
        send_telegram_message(msg)
        last_alert_time[sym] = current_time

def on_message(ws, message):
    data = json.loads(message)['data']
    kline = data['k']
    sym = data['s'].lower()

    if sym not in market_data:
        return

    df = market_data[sym]
    ts = kline['t']
    o, h, l, c, v = float(kline['o']), float(kline['h']), float(kline['l']), float(kline['c']), float(kline['v'])

    if df['timestamp'].iloc[-1] == ts:
        df.iloc[-1] = [ts, o, h, l, c, v]
    else:
        new_row = pd.DataFrame([[ts, o, h, l, c, v]], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        market_data[sym] = pd.concat([df.iloc[1:], new_row], ignore_index=True)

    check_signal(sym)

def start_websocket():
    load_initial_history()
    streams = "/".join([f"{s}@kline_{INTERVAL}" for s in SYMBOLS])
    ws_url = f"wss://data-stream.binance.vision/stream?streams={streams}"
    
    while True:
        try:
            ws = websocket.WebSocketApp(ws_url, on_message=on_message)
            ws.run_forever()
        except Exception as e:
            print(f"WebSocket reconnecting: {e}")
            time.sleep(5)

if __name__ == "__main__":
    threading.Thread(target=start_websocket, daemon=True).start()
    run_web_server()
