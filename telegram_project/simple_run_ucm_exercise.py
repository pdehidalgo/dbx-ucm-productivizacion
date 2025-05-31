import logging
import os
import requests

import telebot
from pydub import AudioSegment
import time 

from config import get_config
from factory.tts_factory import TTSFactory
from speech_to_text import OpenAIWhispersVoiceHandler
from text_to_speech import OpenAITTSHandler

from openai import OpenAI

from system_prompt import SYSTEM_PROMPT_MY_EXAMPLE

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

conversations = {}

client = OpenAI(api_key=config["OPENAI_API_KEY"])
# TODO: Azure OpenAI client


#####################
## SETUP ENVIRON  ###
#####################

# Descarga el fichero de poetry y genera un entorno para instalar las dependencias necesarias
# Crea un fichero de environment .env que vaya a incluir los API tokens necesarios y cárgalo como hemos visto en clase


bot = telebot.TeleBot(
    # Fill
)

#####################
##### INTERFACES  ###
#####################


#####################
####### TEXTO  ######
#####################


@bot.message_handler(func=lambda message: True)
def on_text(message):
    handle_text_message(bot, message)

# Crea una función que dado un menssage del usuario: 
    
def handle_text_message(bot, message):

    # Ex1: Revise que nuestro ID es válido y está en la WHITE LIST

    # Guarde el estado de la conversación y genere una respuesta apropiada
    response = conversation_tracking(message.from_user.id, message.text)

    # Opcional: Sabiendo que podemos realizar acciones sobre el chat de telegram como la siguiente: 
    # bot.send_chat_action(chat_id, "typing")
    # Implementa con tu equipo una función que haga que escriba en base al número de palabras 

    # Envíe la respuesta al usuario
    bot.send_message(message.chat.id, response)



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

    ## Incluye tu código de llamada aquí 
    chat_completion = client....
    return chat_completion.choices[0].message.content


#####################
######## VOZ  #######
#####################


# 1. Crea un TranscribeVoiceHandler que obligue a que el resto de VoiceHandlers tengan la misma estructura
from abc import ABC, abstractmethod

class TranscribeVoiceHandler(ABC):
    @abstractmethod
    def transcribing_voice(self):
        pass

# 2. Basado en la documentación de OpenAI, genera un código que permita transcribir el audio a texto

class LocalWhispersVoiceHandler(TranscribeVoiceHandler):
    def transcribing_voice(self):
        raise 


class OpenAIWhispersVoiceHandler(TranscribeVoiceHandler):

    def __init__(self) -> None:
        self.client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

    def transcribing_voice(self) -> str:

        file_path = "" # Puede que el tema de los paths genere algún error. 

        # sound = AudioSegment.from_file("voice_message.ogg", format="ogg")
        sound = AudioSegment.from_file(f"{file_path}voice_message.ogg", format="ogg")
        sound.export("voice_message.mp3", format="mp3")
        sound.export("voice_message.wav", format="wav")

        try:
            # Tu bloque de código va aquí. 
            try: 
                audio_file = open("voice_message.mp3", "rb")
            except Exception as e: 
                logging.error(f'error al accer al fichero{e.args}')

            transcription = self.client.audio.transcribing.create(
                model="whisper-1", file=audio_file
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
        f"https://api.telegram.org/file/bot{config['TELEGRAM_BOT_TOKEN']}/{file_info.file_path}"
    )

    with open("voice_message.ogg", "wb") as f:
        f.write(file.content)

    text = transcription_handler.transcribing_voice()
    response = conversation_tracking(user_id, text)

    bot.reply_to(message, response)
    
    # Opcional: crea un OpenAITTSHandler basado en TTSHandler como ya habéis hecho previamente. 

    # class TTSHandler...

    # if config["TEXT_TO_SPEECH"]:
    #     voice = tts_handler.generating_voice(response)
    #     if voice:
    #         bot.send_voice(user_id, voice)
    #         voice.close()
    #     try:
    #         os.remove("voice_message_replay.ogg")
    #         os.remove("voice_message.mp3")
    #     except PermissionError:
    #         logging.warning("Could not remove voice files (in use)")



@bot.message_handler(content_types=["voice"])
def on_voice(message):
    handle_voice_message_simple(bot, message, transcription_handler)


@bot.message_handler(func=lambda message: True)
def on_text(message):
    handle_text_message(bot, message)


#####################
##### FICHEROS  #####
#####################

# Implementa el handler para manejar ficheros. 
@bot.message_handler(....)
def on_files(message):
    handle_files(bot, message)


if __name__ == "__main__":
    logging.info("Starting Telegram bot...")
    bot.polling()
