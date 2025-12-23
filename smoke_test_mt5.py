import logging
import os
import sys
import time
from typing import Optional

from dotenv import load_dotenv
import MetaTrader5 as mt5

# Minimal MT5 smoke test.
# Opens a tiny market order on XAUUSD and then closes it immediately.

MAGIC_NUMBER = 234000
FIXED_VOLUME = 0.01
DEVIATION = 20
DEFAULT_SYMBOL = "XAUUSD"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def ensure_mt5_connection() -> bool:
    """Connect and log in to MT5 using env credentials."""
    if not mt5.initialize():
        logger.error("MT5 initialize() failed: %s", mt5.last_error())
        return False

    login_raw = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")

    if not login_raw or not password or not server:
        logger.error("Missing MT5_LOGIN/MT5_PASSWORD/MT5_SERVER in environment")
        return False

    try:
        login = int(login_raw)
    except ValueError:
        logger.error("MT5_LOGIN must be an integer; got %s", login_raw)
        return False

    account_info = mt5.account_info()
    if account_info and account_info.login == login:
        return True

    if not mt5.login(login=login, password=password, server=server):
        logger.error("MT5 login failed: %s", mt5.last_error())
        return False

    return True


def place_market_order(symbol: str, side: str) -> Optional[int]:
    """Place a small market order and return the ticket number."""
    symbol = symbol.upper()
    side = side.lower()

    if side not in {"buy", "sell"}:
        logger.error("Side must be 'buy' or 'sell'")
        return None

    if not mt5.symbol_select(symbol, True):
        logger.error("Failed to select symbol %s", symbol)
        return None

    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        logger.error("No tick data for %s", symbol)
        return None

    is_buy = side == "buy"
    price = tick.ask if is_buy else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": FIXED_VOLUME,
        "type": order_type,
        "price": price,
        "deviation": DEVIATION,
        "magic": MAGIC_NUMBER,
        "comment": "SMOKE_TEST",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error("Order failed: %s (retcode %s)", result.comment, result.retcode)
        return None

    logger.info("Opened %s %s @ %s | ticket=%s", side, symbol, price, result.order)
    return result.order


def close_order(ticket: int, symbol: str):
    """Close the position associated with the given ticket."""
    symbol = symbol.upper()
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        logger.warning("No position found for ticket %s; nothing to close", ticket)
        return

    pos = positions[0]
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        logger.error("No tick data for %s; cannot close", symbol)
        return

    close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": pos.volume,
        "type": close_type,
        "position": ticket,
        "price": price,
        "deviation": DEVIATION,
        "magic": MAGIC_NUMBER,
        "comment": "SMOKE_TEST_CLOSE",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error("Close failed: %s (retcode %s)", result.comment, result.retcode)
    else:
        logger.info("Closed ticket %s", ticket)


def main():
    load_dotenv()
    if not ensure_mt5_connection():
        sys.exit(1)

    symbol = os.getenv("TEST_SYMBOL", DEFAULT_SYMBOL)
    side = os.getenv("TEST_SIDE", "buy")

    logger.info("Placing %s on %s (%.2f lots). This is a live market order.", side, symbol, FIXED_VOLUME)
    ticket = place_market_order(symbol, side)
    if ticket is None:
        sys.exit(1)

    # Small delay to let the trade register, then close it to avoid exposure.
    time.sleep(2)
    close_order(ticket, symbol)

    mt5.shutdown()


if __name__ == "__main__":
    main()
