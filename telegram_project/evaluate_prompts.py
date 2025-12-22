import argparse
import logging
import os
from enum import StrEnum
from io import BytesIO

import requests
import telebot

try:
    from telegram_project.mlflow_azure import (
        EXPERIMENT_NAME,
        get_best_prompt,
        predict_image_description,
        run_experiments,
    )
except ModuleNotFoundError:
    from mlflow_azure import (
        EXPERIMENT_NAME,
        get_best_prompt,
        predict_image_description,
        run_experiments,
    )

logger = logging.getLogger(__name__)


TELEGRAM_BOT_WELCOME_MESSAGE = "Hello, nice to meet you!\nI am Adam, an analyst bot specialized in giving you technical support for your day-to-day issues."
TELEGRAM_BOT_HELP_MESSAGE = "How can I help you?"
TELEGRAM_BOT_DIAGNOSIS_PREFIX = "Diagnosis:\n"


class RunMode(StrEnum):
    """Defines the available modes for running this script."""

    RUN_ALL = "run_all"
    RUN_EXPERIMENTS = "run_experiments"
    RUN_BOT = "run_bot"


def get_telegram_bot_client() -> telebot.TeleBot:
    """
    Initializes and returns a Telegram Bot client.

    Raises:
        ValueError: If the TELEGRAM_BOT_TOKEN environment variable is not set.

    Returns:
        telebot.TeleBot: An initialized Telegram Bot client.
    """
    telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not telegram_token:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN environment variable must be set to run the bot."
        )

    bot = telebot.TeleBot(telegram_token, parse_mode=None)
    telebot.logger.setLevel(logging.INFO)
    return bot


def run_bot() -> None:
    """
    Starts and manages the Telegram bot for image description.
    The bot uses the best prompt determined from MLflow experiments.
    """
    try:
        bot = get_telegram_bot_client()
        telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        best_prompt = get_best_prompt([EXPERIMENT_NAME])

        if not best_prompt:
            best_prompt = (
                "Describe in detail what you see in the image, including key visual elements, "
                "visible text, and any clues about the context."
            )
            logger.warning(
                "No best prompt found from MLflow; falling back to default prompt."
            )

        logger.info(f"Using best prompt for the bot: '{best_prompt}'")

        @bot.message_handler(commands=["start", "help"])
        def send_welcome(message):
            """Handles /start and /help commands."""
            user_id = message.chat.id
            bot.send_message(user_id, TELEGRAM_BOT_WELCOME_MESSAGE)
            bot.reply_to(message, TELEGRAM_BOT_HELP_MESSAGE)

        @bot.message_handler(content_types=["photo"])
        def on_photo(message):
            """Handles incoming photo messages, describes them, and sends back the diagnosis."""
            user_id = message.chat.id

            file_id = message.photo[-1].file_id
            file_info = bot.get_file(file_id)

            file_url = f"https://api.telegram.org/file/bot{telegram_token}/{file_info.file_path}"
            response = requests.get(file_url, stream=True, timeout=(5, 20))
            response.raise_for_status()
            image_bytes = BytesIO(response.content)
            image_bytes.mime_type = response.headers.get("Content-Type")

            description = predict_image_description([best_prompt], image_bytes)[0]

            if description:
                bot.send_message(
                    user_id, f"{TELEGRAM_BOT_DIAGNOSIS_PREFIX}{description}"
                )
            else:
                bot.send_message(
                    user_id,
                    "Sorry, I could not generate a description for the image. Please try again.",
                )

        bot.infinity_polling(timeout=10, long_polling_timeout=5)

    except ValueError as e:
        logger.critical(
            f"Bot setup failed: {e}. Please ensure all required environment variables are set."
        )
    except Exception as e:
        logger.critical(f"An unrecoverable error occurred during bot execution: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run image description prompt evaluation or a Telegram bot."
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in RunMode],
        help="Mode of operation: 'run_all' (experiments then bot), 'run_experiments', or 'run_bot'.",
        default=RunMode.RUN_EXPERIMENTS,
    )
    args = parser.parse_args()

    logger.info(f"Starting application in mode: {args.mode}")

    match args.mode:
        case RunMode.RUN_ALL:
            run_experiments()
            run_bot()
        case RunMode.RUN_EXPERIMENTS:
            run_experiments()
        case RunMode.RUN_BOT:
            run_bot()

    logger.info(f"Application finished running in mode: {args.mode}")
