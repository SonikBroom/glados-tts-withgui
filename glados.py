import tkinter as tk
import torch
from utils.tools import prepare_text
from scipy.io.wavfile import write
import os
import winsound
from threading import Thread

def process_input():
    def save_audio():
        user_input = message_entry.get()  # Get text from the message_entry textbox

        if user_input:
            file_name = file_entry.get()

            if not os.path.exists('output'):
                os.makedirs('output')

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            glados = torch.jit.load('models/glados.pt')
            vocoder = torch.jit.load('models/vocoder-gpu.pt', map_location=device)

            x = prepare_text(user_input).to('cpu')

            with torch.no_grad():
                tts_output = glados.generate_jit(x)
                mel = tts_output['mel_post'].to(device)
                audio = vocoder(mel)

                audio = audio.squeeze()
                audio = audio * 32768.0
                audio = audio.cpu().numpy().astype('int16')
                output_file = f"output/{file_name}.wav"

                write(output_file, 22050, audio)
                console_area.insert(tk.END, f"Audio saved as {output_file}\n")
                winsound.PlaySound(output_file, winsound.SND_FILENAME)

    Thread(target=save_audio).start()

root = tk.Tk()
root.title("GlaDOS TTS")

console_area = tk.Text(root, height=10, width=50)
console_area.grid(row=0, column=0, columnspan=2)

tk.Label(root, text="Message: ").grid(row=1, column=0)
message_entry = tk.Entry(root)
message_entry.grid(row=1, column=1)

tk.Label(root, text="Output File Name: ").grid(row=2, column=0)
file_entry = tk.Entry(root)
file_entry.grid(row=2, column=1)

play_button = tk.Button(root, text="Save Audio", command=process_input)
play_button.grid(row=3, column=0, columnspan=2)

root.mainloop()"./output.wav"])
