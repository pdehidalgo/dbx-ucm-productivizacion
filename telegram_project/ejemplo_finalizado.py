import logging
import os
from abc import ABC, abstractmethod
from functools import wraps

import requests
import telebot
from dotenv import load_dotenv
from openai import AzureOpenAI
from pydub import AudioSegment

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

conversations = {}
WHITELIST = [710453674]

endpoint = os.getenv("OPENAI_ENDPOINT")
model_name = "gpt-4o-mini"
deployment = "gpt-4o-mini"

subscription_key = os.getenv("OPENAI_API_KEY")
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

bot = telebot.TeleBot(os.getenv("TELEGRAM_BOT_TOKEN"))


def whitelist_only(func):
    @wraps(func)
    def wrapper(message, *args, **kwargs):
        user_id = message.from_user.id
        if user_id in WHITELIST:
            return func(message, *args, **kwargs)
        else:
            bot.reply_to(message, "Access denied. ❌")

    return wrapper


@bot.message_handler(commands=["start"])
@whitelist_only
def handle_start(message):
    bot.reply_to(message, "Welcome, authorized user! 🎉")


@bot.message_handler(func=lambda message: True)
@whitelist_only
def on_text(message):
    handle_text_message(bot, message)


def handle_text_message(bot, message):
    # Guarde el estado de la conversación y genere una respuesta apropiada
    response = conversation_tracking(message.from_user.id, message.text)

    # Envíe la respuesta al usuario
    bot.send_message(message.chat.id, response)

    # Opcional: Sabiendo que podemos realizar acciones sobre el chat de telegram como la siguiente:
    # bot.send_chat_action(chat_id, "typing")
    # Implementa con tu equipo una función que haga que escriba en base al número de palabras


def conversation_tracking(user_id: int, text_message: str) -> str:
    # Esta función está bastante acoplada, ¿se puede mejorar?
    user_convo = conversations.get(user_id, [])
    user_convo.append({"role": "user", "content": text_message})
    logging.info(f"{user_id} (user): {text_message}")

    response = generate_response_chat(user_convo)
    user_convo.append({"role": "assistant", "content": response})
    logging.info(f"{user_id} (bot): {response}")

    conversations[user_id] = user_convo
    return response


def generate_response_chat(message_list):
    chat_completion = client.chat.completions.create(
        messages=message_list, max_tokens=4096, temperature=1.0, top_p=1.0, model=deployment
    )

    return chat_completion.choices[0].message.content


class TranscribeVoiceHandler(ABC):
    @abstractmethod
    def transcribing_voice(self):
        pass


class LocalWhispersVoiceHandler(TranscribeVoiceHandler):
    def transcribing_voice(self):
        raise


class OpenAIWhispersVoiceHandler(TranscribeVoiceHandler):

    def __init__(self) -> None:
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY_VOICE"),
            api_version=api_version,
            azure_endpoint=endpoint,
        )

    def transcribing_voice(self) -> str:

        file_path = ""  # Puede que el tema de los paths genere algún error.

        # sound = AudioSegment.from_file("voice_message.ogg", format="ogg")
        sound = AudioSegment.from_file(f"{file_path}voice_message.ogg", format="ogg")
        sound.export("voice_message.mp3", format="mp3")
        sound.export("voice_message.wav", format="wav")

        try:
            # Tu bloque de código va aquí.
            audio_file = open("voice_message.mp3", "rb")

            transcription = self.client.audio.transcriptions.create(
                model="whisper",
                file=audio_file,
            )

            return transcription.text
        except Exception as e:
            logging.error(f"Error transcribing with Whispers {e}")
            return "I text you back later..."


# 3. Rellena tu función para manejar eventos de voz.
transcription_handler = OpenAIWhispersVoiceHandler()


@bot.message_handler(content_types=["voice"])
def on_voice(message):
    handle_voice_message_simple(bot, message, transcription_handler)


def handle_voice_message_simple(bot, message, transcription_handler):
    user_id = message.chat.id
    file_info = bot.get_file(message.voice.file_id)
    file = requests.get(
        f"https://api.telegram.org/file/bot{os.getenv('TELEGRAM_BOT_TOKEN')}/{file_info.file_path}"
    )

    with open("voice_message.ogg", "wb") as f:
        f.write(file.content)

    text = transcription_handler.transcribing_voice()
    response = conversation_tracking(user_id, text)

    bot.reply_to(message, response)


if __name__ == "__main__":
    logging.info("Starting Telegram bot...")
    try:
        bot.polling(none_stop=True)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise
