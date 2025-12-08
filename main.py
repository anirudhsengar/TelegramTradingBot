import asyncio
import logging
import os
import json
import re
from collections import deque
from datetime import datetime, timezone, timedelta
from telethon import TelegramClient, events
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import MetaTrader5 as mt5

# --- Configuration & Setup ---
load_dotenv()

# Logging Setup
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Environment Variables
API_ID = os.getenv("APP_ID")
API_HASH = os.getenv("APP_HASH")
GROUP_IDS_RAW = os.getenv("TELEGRAM_GROUP_IDS", "[]")
try:
    TARGET_GROUP_IDS = [int(g) for g in json.loads(GROUP_IDS_RAW)]
except json.JSONDecodeError:
    logger.error("Failed to parse TELEGRAM_GROUP_IDS. Ensure it is a valid JSON array of integers.")
    TARGET_GROUP_IDS = []

ENDPOINT = "https://models.github.ai/inference"
MODEL_NAME = "microsoft/Phi-4"
TOKEN = os.getenv("GITHUB_TOKEN")

# MT5 Credentials
MT5_LOGIN = int(os.getenv("MT5_LOGIN")) if os.getenv("MT5_LOGIN") else None
MT5_PASSWORD = os.getenv("MT5_PASSWORD")
MT5_SERVER = os.getenv("MT5_SERVER")

# Trading Constants
MAGIC_NUMBER = 234000  # Unique identifier for this bot's trades
FIXED_VOLUME = 0.01    # User requirement: 0.01 lots per trade
DEVIATION = 20         # Slippage tolerance in points
STALE_MESSAGE_SECONDS = 120  # Ignore old Telegram messages
MAX_TRACKED_MESSAGES = 500   # Bound dedupe memory

# Initialize Clients
llm_client = ChatCompletionsClient(
    endpoint=ENDPOINT,
    credential=AzureKeyCredential(TOKEN),
)

telegram_client = TelegramClient('session_name', API_ID, API_HASH)
processed_message_ids: deque[str] = deque(maxlen=MAX_TRACKED_MESSAGES)


def _message_age_seconds(event) -> float:
    """Compute message age accounting for Telethon's time offset if the system clock is skewed."""
    msg_time = event.date
    if not msg_time:
        return 0.0
    if msg_time.tzinfo is None:
        msg_time = msg_time.replace(tzinfo=timezone.utc)

    offset = 0
    try:
        sender = getattr(event._client, "_sender", None)
        offset = getattr(sender, "time_offset", 0) or 0
    except Exception:
        offset = 0

    now = datetime.now(timezone.utc) + timedelta(seconds=offset)
    return (now - msg_time).total_seconds(), offset


# --- MetaTrader 5 Logic ---

def ensure_mt5_connection():
    """Ensures MT5 is connected and logged in. Returns True if successful."""
    if not mt5.initialize():
        logger.error(f"MT5 initialize() failed, error code = {mt5.last_error()}")
        return False
    
    # Check if already connected to the correct account
    account_info = mt5.account_info()
    if account_info and account_info.login == MT5_LOGIN:
        return True

    # Try to login
    if not mt5.login(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        logger.error(f"MT5 login failed, error code = {mt5.last_error()}")
        return False
    
    return True

def execute_trade_sync(signal):
    """Synchronous function to execute trade on MT5."""
    if not ensure_mt5_connection():
        return

    symbol = signal.get("symbol", "XAUUSD").upper()
    side = signal.get("side")

    if side not in {"buy", "sell"}:
        logger.error(f"Invalid or missing side for signal: {signal}")
        return
    
    # Ensure symbol is available
    if not mt5.symbol_select(symbol, True):
        logger.error(f"Failed to select symbol {symbol}")
        return

    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        logger.error(f"No tick data for {symbol}; skipping order")
        return

    # Prepare Order
    order_type = mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL
    price = tick.ask if side == "buy" else tick.bid
    if price is None:
        logger.error(f"Missing price for {symbol}; skipping order")
        return

    # Parse SL/TP (Handle 0 or None and validate against current price)
    raw_sl = signal.get("sl")
    raw_tp = signal.get("tp")
    sl = float(raw_sl) if raw_sl not in (None, "", 0, 0.0) else 0.0
    tp = float(raw_tp) if raw_tp not in (None, "", 0, 0.0) else 0.0

    if side == "buy":
        if sl and sl >= price:
            logger.warning(f"SL {sl} is not below buy price {price}; dropping SL")
            sl = 0.0
        if tp and tp <= price:
            logger.warning(f"TP {tp} is not above buy price {price}; dropping TP")
            tp = 0.0
    else:  # sell
        if sl and sl <= price:
            logger.warning(f"SL {sl} is not above sell price {price}; dropping SL")
            sl = 0.0
        if tp and tp >= price:
            logger.warning(f"TP {tp} is not below sell price {price}; dropping TP")
            tp = 0.0

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": FIXED_VOLUME,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": DEVIATION,
        "magic": MAGIC_NUMBER,
        "comment": "AutoTrade via LLM",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    # Send Order
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"Order failed: {result.comment} (Retcode: {result.retcode})")
    else:
        logger.info(f"Order executed: {side} {symbol} @ {price}, Ticket: {result.order}")

def close_trades_sync(symbol="XAUUSD"):
    """Synchronous function to close all trades for a symbol with MAGIC_NUMBER."""
    if not ensure_mt5_connection():
        return

    symbol = symbol.upper()
    positions = mt5.positions_get(symbol=symbol)
    
    if not positions:
        logger.info(f"No open positions found for {symbol}")
        return

    count = 0
    for pos in positions:
        # Only close trades opened by this bot
        if pos.magic != MAGIC_NUMBER:
            continue

        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            logger.error(f"No tick data for {symbol}; cannot close position {pos.ticket}")
            continue

        # Determine close type
        order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": pos.volume,
            "type": order_type,
            "position": pos.ticket,
            "price": price,
            "deviation": DEVIATION,
            "magic": MAGIC_NUMBER,
            "comment": "AutoClose via LLM",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to close position {pos.ticket}: {result.comment}")
        else:
            logger.info(f"Closed position {pos.ticket}")
            count += 1
    
    logger.info(f"Closed {count} positions for {symbol}")

# --- Async Wrappers ---

async def open_position_async(signal):
    """Wraps the blocking MT5 trade execution in a thread."""
    logger.info(f"Processing OPEN signal: {signal}")
    await asyncio.to_thread(execute_trade_sync, signal)

async def close_positions_async(symbol):
    """Wraps the blocking MT5 close execution in a thread."""
    logger.info(f"Processing CLOSE signal for {symbol}")
    await asyncio.to_thread(close_trades_sync, symbol)


# --- Signal Processing ---

def is_potential_trade_text(text: str) -> bool:
    """Fast pre-filter to avoid wasting LLM calls on chatter."""
    if not text: return False
    text_lower = text.lower()
    keywords = ("buy", "sell", "xau", "gold", "xauusd", "tp", "sl", "stop", "close", "profit")
    return any(k in text_lower for k in keywords)

async def parse_trade_signal(text: str):
    """Uses Azure AI Inference to extract structured trade data."""
    prompt = (
        "You are a financial trading assistant. Extract the trading instruction from the text. "
        "Return ONLY a compact JSON object with no markdown formatting. "
        "Schema: {\"action\": \"open|close|none\", "
        "\"symbol\": \"XAUUSD\" (default if Gold/XAU mentioned), "
        "\"side\": \"buy|sell|null\", "
        "\"tp\": number|null, \"sl\": number|null}. "
        "Rules: "
        "1. If text implies closing positions (e.g., 'Close all', 'Close Gold'), set action='close'. "
        "2. If text implies entering a trade, set action='open'. "
        "3. If unclear or just chatter, set action='none'. "
        "4. Convert all prices to pure numbers."
    )
    
    def _extract_json_block(raw: str) -> str:
        # Drop code fences and grab the first {...} block
        cleaned = raw.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        return match.group(0) if match else cleaned

    try:
        response = await asyncio.to_thread(
            llm_client.complete,
            messages=[SystemMessage(prompt), UserMessage(f"Message: {text}")],
            temperature=0.1,
            top_p=0.9,
            max_tokens=200,
            model=MODEL_NAME
        )

        content = response.choices[0].message.content or ""
        json_payload = _extract_json_block(content)
        candidate = json.loads(json_payload)

        action = candidate.get("action")
        side = candidate.get("side")
        symbol = (candidate.get("symbol") or "XAUUSD").upper()

        if action not in {"open", "close", "none"}:
            raise ValueError(f"Invalid action {action}")
        if action == "open" and side not in {"buy", "sell"}:
            raise ValueError(f"Invalid side {side} for open action")

        return {
            "action": action,
            "symbol": symbol,
            "side": side if action == "open" else None,
            "tp": candidate.get("tp"),
            "sl": candidate.get("sl"),
        }
    except Exception as e:
        logger.error(f"LLM Parse Error: {e} | raw content: {content if 'content' in locals() else text}")
        return {"action": "none"}


# --- Telegram Event Handler ---

async def handler(event):
    message_id = f"{event.chat_id}:{event.id}"
    # Ignore stale messages to avoid replayed trades (adjusted for Telethon time offset)
    age, offset = _message_age_seconds(event)
    if age > STALE_MESSAGE_SECONDS:
        # If the server thinks our clock is badly skewed, don't drop; log and continue
        if abs(offset) > 300:
            logger.warning(
                f"Clock skew detected (offset={offset}s); overriding stale guard for {message_id} age={age}s"
            )
        else:
            logger.info(f"Skipping stale message {message_id} age={age}s (offset-adjusted)")
            return

    if message_id in processed_message_ids:
        logger.debug(f"Duplicate message {message_id} ignored")
        return
    processed_message_ids.append(message_id)

    sender = await event.get_sender()
    sender_name = getattr(sender, 'first_name', 'Unknown')
    text = event.text or ""
    
    logger.info(f"New Message from {sender_name}: {text}")
    
    if not is_potential_trade_text(text):
        return

    signal = await parse_trade_signal(text)
    action = signal.get("action")
    
    if action == "open":
        await open_position_async(signal)
    elif action == "close":
        await close_positions_async(signal.get("symbol", "XAUUSD"))
    else:
        logger.debug("No actionable trade signal detected.")

# --- Main Entry Point ---

if __name__ == "__main__":
    print("Starting Telegram Trading Bot...")
    print(f"Listening to groups: {TARGET_GROUP_IDS}")

    if not TARGET_GROUP_IDS:
        raise SystemExit("No TELEGRAM_GROUP_IDS configured. Aborting to avoid listening to all chats.")

    # Initial connection check
    if ensure_mt5_connection():
        print("Connected to MetaTrader 5 successfully.")
    else:
        print("WARNING: Failed to connect to MetaTrader 5. Will retry on first signal.")

    telegram_client.add_event_handler(handler, events.NewMessage(chats=TARGET_GROUP_IDS))

    try:
        telegram_client.start()
        telegram_client.run_until_disconnected()
    finally:
        mt5.shutdown()