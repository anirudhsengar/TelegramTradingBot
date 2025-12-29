import asyncio
import logging
import os
import json
import re
from collections import deque
from datetime import datetime, timezone, timedelta
from telethon import TelegramClient, events
from dotenv import load_dotenv
from openai import OpenAI
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

MODEL_NAME = "gpt-4o"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise SystemExit("Missing OPENAI_API_KEY in environment; add it to your .env file.")

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
llm_client = OpenAI(api_key=OPENAI_API_KEY)

telegram_client = TelegramClient('session_name', API_ID, API_HASH)
# Track last seen text and parsed signal per message so we can react to edits safely
message_state: dict[str, dict] = {}
message_order: deque[str] = deque()
recent_messages: dict[int, deque[dict]] = {}


def _message_age_seconds(event) -> tuple[float, int]:
    """Compute message age accounting for Telethon's time offset if the system clock is skewed."""
    msg = getattr(event, "message", None)
    msg_time = getattr(msg, "edit_date", None) or getattr(event, "date", None)
    if not msg_time:
        return 0.0, 0
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


def _remember_message(message_id: str, text: str, signal: dict):
    """Remember the last processed text/signal for a message id and prune old entries."""
    if message_id in message_order:
        message_order.remove(message_id)
    message_order.append(message_id)
    message_state[message_id] = {"text": text, "signal": signal}

    while len(message_order) > MAX_TRACKED_MESSAGES:
        old_id = message_order.popleft()
        message_state.pop(old_id, None)


def _detect_language(text: str) -> str:
    """Very light language hint: detect Arabic characters; default to 'en'."""
    if re.search(r"[\u0600-\u06FF]", text or ""):
        return "ar"
    return "en"


def _remember_recent_message(chat_id: int, sender: str, text: str, timestamp: str):
    """Keep a short rolling window of recent messages per chat for LLM context."""
    window = recent_messages.setdefault(chat_id, deque(maxlen=5))
    window.append({"sender": sender, "text": text, "ts": timestamp})


def _get_recent_messages(chat_id: int, limit: int = 3):
    window = recent_messages.get(chat_id)
    if not window:
        return []
    return list(window)[-limit:]


def _positions_snapshot(symbol_filter: str = "XAUUSD", max_items: int = 5):
    """Lightweight view of this bot's positions for context. Does not modify trades."""
    try:
        positions = mt5.positions_get()
    except Exception:
        return []

    if not positions:
        return []

    snapshot = []
    for pos in positions:
        if pos.magic != MAGIC_NUMBER:
            continue
        if symbol_filter and pos.symbol.upper() != symbol_filter.upper():
            continue
        snapshot.append({
            "ticket": pos.ticket,
            "symbol": pos.symbol,
            "side": "buy" if pos.type == mt5.ORDER_TYPE_BUY else "sell",
            "entry": pos.price_open,
            "sl": pos.sl,
            "tp": pos.tp,
            "volume": pos.volume,
        })
        if len(snapshot) >= max_items:
            break
    return snapshot


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


def update_positions_sync(signal):
    """Update SL/TP on existing positions when a Telegram message is edited."""
    if not ensure_mt5_connection():
        return

    symbol = (signal.get("symbol") or "XAUUSD").upper()
    side = signal.get("side")

    raw_sl = signal.get("sl")
    raw_tp = signal.get("tp")

    if raw_sl in (None, "", 0, 0.0) and raw_tp in (None, "", 0, 0.0):
        logger.info(f"No SL/TP update provided for {symbol}; skipping")
        return

    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        logger.info(f"No open positions found for {symbol} to update")
        return

    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        logger.error(f"No tick data for {symbol}; cannot update positions")
        return

    requested_sl = float(raw_sl) if raw_sl not in (None, "", 0, 0.0) else None
    requested_tp = float(raw_tp) if raw_tp not in (None, "", 0, 0.0) else None

    updated = 0
    for pos in positions:
        if pos.magic != MAGIC_NUMBER:
            continue

        # Respect side if supplied; otherwise update all bot positions for the symbol
        if side == "buy" and pos.type != mt5.ORDER_TYPE_BUY:
            continue
        if side == "sell" and pos.type != mt5.ORDER_TYPE_SELL:
            continue

        current_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask

        target_sl = pos.sl if requested_sl is None else requested_sl
        target_tp = pos.tp if requested_tp is None else requested_tp

        if pos.type == mt5.ORDER_TYPE_BUY:
            if target_sl and target_sl >= current_price:
                logger.warning(f"SL {target_sl} not below current price {current_price} for BUY; keeping existing {pos.sl}")
                target_sl = pos.sl
            if target_tp and target_tp <= current_price:
                logger.warning(f"TP {target_tp} not above current price {current_price} for BUY; keeping existing {pos.tp}")
                target_tp = pos.tp
        else:
            if target_sl and target_sl <= current_price:
                logger.warning(f"SL {target_sl} not above current price {current_price} for SELL; keeping existing {pos.sl}")
                target_sl = pos.sl
            if target_tp and target_tp >= current_price:
                logger.warning(f"TP {target_tp} not below current price {current_price} for SELL; keeping existing {pos.tp}")
                target_tp = pos.tp

        if target_sl == pos.sl and target_tp == pos.tp:
            logger.info(f"Position {pos.ticket} already has requested SL/TP; skipping")
            continue

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": pos.ticket,
            "symbol": symbol,
            "sl": target_sl,
            "tp": target_tp,
            "deviation": DEVIATION,
            "magic": MAGIC_NUMBER,
            "comment": "Update SL/TP via edit",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to update position {pos.ticket}: {result.comment}")
        else:
            logger.info(f"Updated position {pos.ticket} SL={target_sl} TP={target_tp}")
            updated += 1

    if updated == 0:
        logger.info(f"No positions were updated for {symbol}")
    else:
        logger.info(f"Updated {updated} position(s) for {symbol}")

# --- Async Wrappers ---

async def open_position_async(signal):
    """Wraps the blocking MT5 trade execution in a thread."""
    logger.info(f"Processing OPEN signal: {signal}")
    await asyncio.to_thread(execute_trade_sync, signal)

async def close_positions_async(symbol):
    """Wraps the blocking MT5 close execution in a thread."""
    logger.info(f"Processing CLOSE signal for {symbol}")
    await asyncio.to_thread(close_trades_sync, symbol)


async def update_positions_async(signal):
    """Wraps SL/TP updates in a thread for edited Telegram messages."""
    logger.info(f"Processing UPDATE signal: {signal}")
    await asyncio.to_thread(update_positions_sync, signal)


# --- Signal Processing ---

def is_potential_trade_text(text: str) -> bool:
    """Fast pre-filter to avoid wasting LLM calls on chatter."""
    if not text: return False
    text_lower = text.lower()
    keywords = ("buy", "sell", "xau", "gold", "xauusd", "tp", "sl", "stop", "close", "profit")
    return any(k in text_lower for k in keywords)

async def parse_trade_signal(text: str, context: dict):
    """Uses OpenAI GPT-4o to extract structured trade data with contextual hints."""
    prompt = (
        "You are a financial trading assistant. Extract the trading instruction from the provided JSON context. "
        "Return ONLY a compact JSON object with no markdown formatting. "
        "Schema: {\"action\": \"open|close|none\", "
        "\"symbol\": \"XAUUSD\" (default if Gold/XAU mentioned), "
        "\"side\": \"buy|sell|null\", "
        "\"tp\": number|null, \"sl\": number|null}. "
        "Rules: "
        "1. If text implies closing positions (e.g., 'Close all', 'Close Gold'), set action='close'. "
        "2. If text implies entering a trade, set action='open'. "
        "3. If unclear or just chatter, set action='none'. "
        "4. Convert all prices to pure numbers. "
        "5. Use language_hint to interpret the text. "
        "6. Use recent_messages for follow-ups like updated TP/SL. "
        "7. If positions show an existing bot trade on the symbol and the new text updates TP/SL, prefer action='open' with new tp/sl (the executor will update)."
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
        user_payload = json.dumps(context, ensure_ascii=False)
        response = await asyncio.to_thread(
            llm_client.chat.completions.create,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Context JSON:\n{user_payload}"},
            ],
            temperature=0.1,
            top_p=0.9,
            max_tokens=200,
            model=MODEL_NAME,
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
    is_edit = isinstance(event, events.MessageEdited.Event)

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

    sender = await event.get_sender()
    sender_name = getattr(sender, 'first_name', 'Unknown')
    text = event.text or ""

    logger.info(f"{'Edited' if is_edit else 'New'} Message from {sender_name}: {text}")

    last_state = message_state.get(message_id)
    if last_state and last_state.get("text") == text:
        logger.debug(f"Message {message_id} text unchanged; ignoring")
        return

    if not is_potential_trade_text(text):
        _remember_message(message_id, text, {"action": "none"})
        _remember_recent_message(event.chat_id, sender_name, text, (event.message.edit_date or event.date).isoformat() if event.date else "")
        return

    context = {
        "message": text,
        "sender": {"name": sender_name, "id": getattr(sender, "id", None)},
        "chat": {"id": event.chat_id, "type": "channel" if getattr(event, "is_channel", False) else "chat"},
        "language_hint": _detect_language(text),
        "recent_messages": _get_recent_messages(event.chat_id),
        "positions": _positions_snapshot(symbol_filter="XAUUSD"),
        "defaults": {
            "default_symbol": "XAUUSD",
            "fixed_volume": FIXED_VOLUME,
            "magic_number": MAGIC_NUMBER,
            "allowed_actions": ["open", "close", "update"],
            "slippage_points": DEVIATION,
        },
        "time": {
            "timestamp": (event.message.edit_date or event.date).isoformat() if event.date else "",
            "message_age_seconds": age,
            "stale_after_seconds": STALE_MESSAGE_SECONDS,
            "time_offset_seconds": offset,
        },
    }

    signal = await parse_trade_signal(text, context)
    action = signal.get("action")
    prev_signal = last_state.get("signal") if last_state else None

    if is_edit and prev_signal and prev_signal.get("action") == "open" and action == "open":
        await update_positions_async(signal)
    elif action == "open":
        await open_position_async(signal)
    elif action == "close":
        await close_positions_async(signal.get("symbol", "XAUUSD"))
    else:
        logger.debug("No actionable trade signal detected.")

    _remember_message(message_id, text, signal)
    _remember_recent_message(event.chat_id, sender_name, text, context["time"]["timestamp"])

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
    telegram_client.add_event_handler(handler, events.MessageEdited(chats=TARGET_GROUP_IDS))

    try:
        telegram_client.start()
        telegram_client.run_until_disconnected()
    finally:
        mt5.shutdown()