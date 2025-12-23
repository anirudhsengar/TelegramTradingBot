# TelegramTradingBot

Telegram-driven MetaTrader 5 executor that listens to curated Telegram groups, uses the GitHub Models Phi-4 (via Azure AI Inference) to parse trade instructions, and places/updates/ closes XAUUSD trades automatically with fixed 0.01-lot sizing.

## What it does
- Listens to specific Telegram chats only (JSON array in `TELEGRAM_GROUP_IDS`).
- Filters likely trade messages, asks the LLM to extract `{action, symbol, side, tp, sl}`.
- Executes `open` and `close` instructions on MT5; edits update SL/TP for existing bot trades.
- Ignores stale messages (>120s) and deduplicates repeated text.
- Keeps a small context window of recent chat messages and current bot positions for better LLM parsing.

## Prerequisites
- Python 3.10+.
- MetaTrader 5 installed on the same machine and logged into the target broker account.
- Telegram API credentials from https://my.telegram.org (API ID/Hash).
- GitHub Models access token with permissions for `models.github.ai` endpoint.

## Setup
1) Clone / open the repo.
2) Create a virtual env and install deps:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
3) Create a `.env` file (same folder as `main.py`):
   ```dotenv
   APP_ID=123456
   APP_HASH=your_telegram_api_hash
   TELEGRAM_GROUP_IDS=[123456789,-987654321]  # JSON array of chat IDs to monitor
   GITHUB_TOKEN=ghp_your_token_for_models

   MT5_LOGIN=12345678
   MT5_PASSWORD=your_mt5_password
   MT5_SERVER=Broker-Server

   # Optional helpers
   PHONE_NUMBER=+10001112222      # used by get_ids.py when listing dialogs
   TEST_SYMBOL=XAUUSD             # overrides default for smoke test
   TEST_SIDE=buy                  # buy|sell for smoke test
   ```
4) (Recommended) Create a Telethon session once so the bot can sign in without SMS prompts:
   ```powershell
   python get_ids.py
   ```
   Follow the interactive prompt; this also prints your dialog IDs for `TELEGRAM_GROUP_IDS`.

## Running the bot
```powershell
python main.py
```
- On startup it validates `TELEGRAM_GROUP_IDS`; if empty, the bot aborts to avoid listening to all chats.
- When a message contains trading cues (buy/sell/gold/xau/sl/tp/close), the LLM parses it.
- Actions:
  - `open`: places a 0.01-lot market order on the parsed symbol (default `XAUUSD`).
  - `close`: closes all bot-tagged positions for that symbol.
  - Edited messages with new TP/SL update existing bot positions (same symbol, same side where relevant).
- Trades are tagged with magic number `234000` and comment `AutoTrade via LLM`.

## Smoke test (MT5 connectivity)
A minimal live-order round-trip to verify MT5 credentials and symbol availability:
```powershell
python smoke_test_mt5.py
```
It opens a tiny market order on `TEST_SYMBOL` (default `XAUUSD`) and closes it after ~2 seconds. Use only on accounts where this is safe.

## Behavior notes
- Fixed volume: `0.01` lots; slippage tolerance: `20` points.
- Stale guard: messages older than 120 seconds (after time-offset correction) are skipped.
- Position snapshot: only bot-tagged positions (magic `234000`) are considered when building context or updating TP/SL.
- Language hint: basic Arabic detection to help the LLM interpret content.

## Safe-use reminders
- This script can place live trades. Use demo accounts first.
- Keep system time synced; large clock skew may affect stale detection.
- Protect your `.env` and `session_name.session` files; they contain credentials/tokens.

## Files
- `main.py` – Telegram listener, LLM-based signal parser, MT5 executor.
- `get_ids.py` – interactive helper to print your dialog/chat IDs and create the Telethon session.
- `smoke_test_mt5.py` – minimal MT5 round-trip sanity check.
- `requirements.txt` – runtime dependencies.
