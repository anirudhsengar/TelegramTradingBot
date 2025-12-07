from telethon import TelegramClient, events
from dotenv import load_dotenv
import os
import json
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import MetaTrader5 as mt5

load_dotenv()

api_id = os.getenv("APP_ID")
api_hash = os.getenv("APP_HASH")
group_ids_raw = os.getenv("TELEGRAM_GROUP_IDS", "[]")
target_group_ids = [int(g) for g in json.loads(group_ids_raw)]

endpoint = "https://models.github.ai/inference"
model_name = "microsoft/Phi-4"
token = os.getenv("GITHUB_TOKEN")
mt5_login = os.getenv("MT5_LOGIN")
mt5_password = os.getenv("MT5_PASSWORD")
mt5_server = os.getenv("MT5_SERVER")

llm_client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

def init_mt5():
    if mt5 is None:
        print("MetaTrader5 package not installed.")
        return False
    if not mt5.initialize():
        print("MT5 initialize() failed")
        return False
    if not mt5.login(login=int(mt5_login), password=mt5_password, server=mt5_server):
        print("MT5 login failed")
        return False
    return True

def is_potential_trade_text(text: str) -> bool:
    text_lower = text.lower()
    keywords = ("buy", "sell", "xau", "gold", "tp", "sl", "stop", "close")
    return any(k in text_lower for k in keywords)

def parse_trade_signal(text: str):
    prompt = (
        "You extract a trading instruction from the text. "
        "Return compact JSON: {\"action\": \"open|close\", "
        "\"symbol\": \"XAUUSD\", \"side\": \"buy|sell|null\", "
        "\"entry_min\": number|null, \"entry_max\": number|null, "
        "\"tp\": number|null, \"sl\": number|null}. "
        "If no trade, return {\"action\":\"none\"}. "
        "Prefer take-profit 1 if multiple are present."
    )
    resp = llm_client.complete(
        messages=[
            SystemMessage(prompt),
            UserMessage(f"Message: {text}")
        ],
        temperature=0.1,
        top_p=0.9,
        max_tokens=200,
        model=model_name
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"LLM parse error: {e}")
        return {"action": "none"}

def open_position(signal):
    if not init_mt5():
        return
    symbol = signal.get("symbol", "XAUUSD")
    side = signal.get("side")
    entry = signal.get("entry_min") or signal.get("entry_max")
    tp = signal.get("tp")
    sl = signal.get("sl")
    if side not in ("buy", "sell"):
        print("Missing side for trade.")
        return
    mt5.symbol_select(symbol, True)
    order_type = mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": 0.1,
        "type": order_type,
        "price": mt5.symbol_info_tick(symbol).ask if side == "buy" else mt5.symbol_info_tick(symbol).bid,
        "sl": sl or 0.0,
        "tp": tp or 0.0,
        "deviation": 20,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    if entry:
        request["price"] = float(entry)
    result = mt5.order_send(request)
    print(f"Open result: {result}")

def close_positions(symbol="XAUUSD"):
    if not init_mt5():
        return
    positions = mt5.positions_get(symbol=symbol)
    if positions:
        for pos in positions:
            order_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).bid if order_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(symbol).ask
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": pos.volume,
                "type": order_type,
                "position": pos.ticket,
                "price": price,
                "deviation": 20,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            print(f"Close result: {result}")
    else:
        print("No open positions to close.")

# Create the client
client = TelegramClient('session_name', api_id, api_hash)

print("Client Created. Listening for messages...")

# Define the event handler
# This checks if the message comes from the specific chat IDs
@client.on(events.NewMessage(chats=target_group_ids))
async def handler(event):
    sender = await event.get_sender()
    sender_name = getattr(sender, 'first_name', 'Unknown')
    text = event.text or ""
    print(f"New Message from {sender_name}: {text}")
    if not is_potential_trade_text(text):
        return
    signal = parse_trade_signal(text)
    action = signal.get("action")
    if action == "open":
        open_position(signal)
    elif action == "close":
        close_positions(signal.get("symbol", "XAUUSD"))
    else:
        print("No trade action taken.")

# Start the client
client.start()
client.run_until_disconnected()