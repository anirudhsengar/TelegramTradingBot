from telethon.sync import TelegramClient
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

api_id = os.getenv("APP_ID")
api_hash = os.getenv("APP_HASH")
phone_number = os.getenv("PHONE_NUMBER")  # Format: +1234567890

async def main():
    client = TelegramClient('session_name', api_id, api_hash)
    await client.start(phone=phone_number)  # will prompt for code if first run
    async with client:
        async for dialog in client.iter_dialogs():
            print(f"Name: {dialog.name} | ID: {dialog.id}")

if __name__ == "__main__":
    asyncio.run(main())