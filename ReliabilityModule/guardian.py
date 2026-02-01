import logging
import os
import json
import codecs
import sys
from logging.handlers import RotatingFileHandler
from discord_webhook import DiscordWebhook, DiscordEmbed
from datetime import datetime
from dotenv import load_dotenv

# Load .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# Set up stdout to handle UTF-8 if it's not already
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Configure Logging (Rotates files so you don't run out of disk space)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Ensure file handler uses UTF-8
log_handler = RotatingFileHandler('bot_audit.log', maxBytes=5*1024*1024, backupCount=5, encoding='utf-8')
log_handler.setFormatter(log_formatter)

logger = logging.getLogger("OneBot")
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)

# Create a stream handler that enforces UTF-8 for the console
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

def send_alert(title: str, message: str, color: str = "ff0000"):
    """
    Sends a notification to your phone via Discord.
    """
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        logger.warning("DISCORD_WEBHOOK_URL not found in environment variables. Notification skipped.")
        return

    webhook = DiscordWebhook(url=webhook_url)
    embed = DiscordEmbed(title=title, description=message, color=color)
    embed.set_timestamp()
    webhook.add_embed(embed)
    
    try:
        webhook.execute()
    except Exception as e:
        logger.error(f"Failed to send alert: {e}")

def log_trade(action, amount, price, reason):
    """
    Structured JSON logging for auditability.
    """
    entry = {
        "event": "TRADE_EXECUTION",
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "amount": amount,
        "price": price,
        "reason": reason
    }
    logger.info(json.dumps(entry))