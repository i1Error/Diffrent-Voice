import io
import requests
from pydub import AudioSegment
from pydub.playback import play
import speech_recognition as sr
import queue
import tempfile
import whisper
import threading
import click
import numpy as np
import json

EL_key = "EL_key"
EL_voice = "EL_voice"

@click.command()
@click.option("--model", default="base", help="Model to use", type=click.Choice(["tiny","base", "small","medium","large"]))
@click.option("--english", default=False, help="Whether to use English model",is_flag=True, type=bool)
@click.option("--german", default=False, help="Whether to use German model",is_flag=True, type=bool)
@click.option("--verbose", default=False, help="Whether to print verbose output", is_flag=True,type=bool)
@click.option("--energy", default=300, help="Energy level for mic to detect", type=int)
@click.option("--dynamic_energy", default=False,is_flag=True, help="Flag to enable dynamic engergy", type=bool)
@click.option("--pause", default=0.8, help="Pause time before entry ends", type=float)
@click.option("--save_file",default=False, help="Flag to save file", is_flag=True,type=bool)

def main(model, english, german, verbose, energy, pause, dynamic_energy, save_file):
    temp_dir = tempfile.mkdtemp() if save_file else None
    if german:
        model = model + ".de"
    elif model != "large" and english:
        model = model + ".en"
    audio_model = whisper.load_model(model)
    audio_queue = queue.Queue()
    result_queue = queue.Queue()
    threading.Thread(target=record_audio, args=(audio_queue, energy, pause, dynamic_energy, save_file, temp_dir)).start()
    threading.Thread(target=transcribe_forever, args=(audio_queue, result_queue, audio_model, english, verbose, save_file)).start()
    while True:
        whisper_result = result_queue.get()
        print(whisper_result)

def play_tts_result(tts_result):
    audio_content = AudioSegment.from_file(io.BytesIO(tts_result), format="mp3")
    play(audio_content)

def record_audio(audio_queue, energy, pause, dynamic_energy, save_file, temp_dir):
    r = sr.Recognizer()
    r.energy_threshold = energy
    r.pause_threshold = pause
    r.dynamic_energy_threshold = dynamic_energy

    with sr.Microphone(sample_rate=16000) as source:
        print("Say something!")
        i = 0
        while True:
            audio = r.listen(source)
            if save_file:
                data = io.BytesIO(audio.get_wav_data())
                audio_clip = AudioSegment.from_file(data)
                filename = os.path.join(temp_dir, f"temp{i}.wav")
                audio_clip.export(filename, format="wav")
                audio_data = filename
            else:
                torch_audio = torch.from_numpy(np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0)
                audio_data = torch_audio

            audio_queue.put_nowait(audio_data)
            i += 1

def tts(message):
    url = f'https://api.elevenlabs.io/v1/text-to-speech/{EL_voice}'
    headers = {
        'accept': 'audio/mpeg',
        'xi-api-key': EL_key,
        'Content-Type': 'application/json'
    }
    data = {
        'text': message,
        'voice_settings': {
            'stability': 0.75,
            'similarity_boost': 0.75
        }
    }
    response = requests.post(url, headers=headers, json=data)
    return response.content

def transcribe_forever(audio_queue, result_queue, audio_model, english, verbose, save_file):
    while True:
        audio_data = audio_queue.get()
        if english:
            result = audio_model.transcribe(audio_data, language='english')
        else:
            result = audio_model.transcribe(audio_data)
        if not verbose:
            predicted_text = result["text"]
            result_queue.put_nowait("You said: " + predicted_text)
            tts_result = tts(predicted_text)
            play_tts_result(tts_result)
        else:
            result_queue.put_nowait(result)

        if save_file:
            os.remove(audio_data)

main()
