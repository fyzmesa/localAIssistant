import tkinter as tk
import sounddevice as sd
from scipy.io.wavfile import write, read
import threading
import os
import numpy as np
import time
from gpt4all import GPT4All
import whisper
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

whisper_model = whisper.load_model("base")
model = GPT4All("/home/USERNAME/.local/share/nomic.ai/GPT4All/Qwen2-7B-LLM-Q4_K_M.gguf")
tts_model = ChatterboxTTS.from_pretrained(device="cuda")

fs = 44100
duration = 120
audio_data = []
is_recording = False
record_start_time = None

output_dir = "/home/USERNAME/DEV/python/output"
os.makedirs(output_dir, exist_ok=True)
audio_path = os.path.join(output_dir, "audio.wav")
text_path = os.path.join(output_dir, "transcript.txt")
tts_audio_path = os.path.join(output_dir, "tts.wav")

def update_clock():
    if is_recording and record_start_time is not None:
        elapsed = int(time.time() - record_start_time)
        clock_label.config(text=f"Recording: {elapsed}s")
        root.after(1000, update_clock)
    else:
        clock_label.config(text="")

def record_audio():
    global is_recording, audio_data, record_start_time
    is_recording = True
    audio_data.clear()
    record_start_time = time.time()
    record_button.config(bg="red", activebackground="red")
    update_clock()

    def callback(indata, frames, time_info, status):
        if is_recording:
            audio_data.append(indata.copy())

    with sd.InputStream(samplerate=fs, channels=1, callback=callback):
        sd.sleep(int(duration * 1000))

def stop_and_transcribe():
    global is_recording, gpt4all_answer
    is_recording = False
    record_button.config(bg="#d9d9d9", activebackground="#d9d9d9")
    time.sleep(1)

    if audio_data:
        audio_np = np.concatenate(audio_data, axis=0)
        write(audio_path, fs, audio_np)

        try:
            result = whisper_model.transcribe(audio_path)
            transcription = result["text"]

            text_output.delete("1.0", tk.END)
            text_output.insert(tk.END, transcription)

            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(transcription)

            gpt4all_answer_text.delete("1.0", tk.END)
            gpt4all_answer_text.insert(tk.END, "Generating answer...")

            def gpt4all_thread():
                global gpt4all_answer
                with model.chat_session():
                    gpt4all_answer = model.generate(transcription, max_tokens=512)
                gpt4all_answer_text.delete("1.0", tk.END)
                gpt4all_answer_text.insert(tk.END, gpt4all_answer)
                synthesize_tts(gpt4all_answer)
            threading.Thread(target=gpt4all_thread).start()

        except Exception as e:
            text_output.delete("1.0", tk.END)
            text_output.insert(tk.END, f"Transcription failed: {e}")
            gpt4all_answer_text.delete("1.0", tk.END)
            gpt4all_answer_text.insert(tk.END, "")
    else:
        text_output.delete("1.0", tk.END)
        text_output.insert(tk.END, "No audio recorded.")
        gpt4all_answer_text.delete("1.0", tk.END)
        gpt4all_answer_text.insert(tk.END, "")

def synthesize_tts(answer_text):
    try:
        wav = tts_model.generate(answer_text)
        ta.save(tts_audio_path, wav, tts_model.sr)
    except Exception as e:
        print(f"TTS synthesis failed: {e}")

def play_tts():
    try:
        wav, sr = ta.load(tts_audio_path)
        sd.play(wav.squeeze().numpy(), sr)
        sd.wait()
    except Exception as e:
        print(f"Playback failed: {e}")

def copy_transcript():
    transcript = text_output.get("1.0", tk.END)
    root.clipboard_clear()
    root.clipboard_append(transcript)

root = tk.Tk()
root.title("Voice Transcriber")
root.geometry("400x600")

record_button = tk.Button(root, text="Record", command=lambda: threading.Thread(target=record_audio).start())
record_button.pack(pady=10)

stop_button = tk.Button(root, text="Stop & Transcribe", command=lambda: threading.Thread(target=stop_and_transcribe).start())
stop_button.pack(pady=10)

clock_label = tk.Label(root, text="", font=("Helvetica", 12))
clock_label.pack(pady=5)

text_output = tk.Text(root, wrap=tk.WORD, height=8)
text_output.pack(pady=10)

copy_button = tk.Button(root, text="Copy 📋", command=copy_transcript)
copy_button.pack(pady=5)

gpt4all_label = tk.Label(root, text="GPT4All Answer:", font=("Helvetica", 12))
gpt4all_label.pack(pady=5)

gpt4all_answer_text = tk.Text(root, wrap=tk.WORD, height=8)
gpt4all_answer_text.pack(pady=10)

play_button = tk.Button(root, text="Play Answer 🔊", command=lambda: threading.Thread(target=play_tts).start())
play_button.pack(pady=5)

root.mainloop()
