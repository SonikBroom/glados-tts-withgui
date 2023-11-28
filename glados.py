import tkinter as tk
import torch
from utils.tools import prepare_text
from scipy.io.wavfile import write
import os
import pyaudio
from threading import Thread

def process_input():
    def save_audio():
        user_input = message_entry.get()

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

                selected_device_index = int(output_device_var.get().split()[0])  # Extract device index
                play_on_default = broadcast_default_var.get()
                play_on_selected = broadcast_selected_var.get()

                thread_selected = Thread(target=play_audio_on_device, args=(audio, selected_device_index, play_on_selected))
                thread_default = Thread(target=play_audio_on_device, args=(audio, None if play_on_default else selected_device_index, play_on_default))
                thread_selected.start()
                if play_on_default:
                    thread_default.start()

    def play_audio_on_device(audio_data, device_index=None, play=False):
        if play:
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=22050,
                            output=True,
                            output_device_index=device_index)
            stream.write(audio_data.tobytes())
            stream.stop_stream()
            stream.close()
            p.terminate()

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

output_devices = pyaudio.PyAudio()
output_device_names = [f"{i} {output_devices.get_device_info_by_index(i)['name']}" for i in range(output_devices.get_device_count()) if output_devices.get_device_info_by_index(i)['maxOutputChannels'] > 0]
output_device_var = tk.StringVar(root)
output_device_var.set(output_device_names[0])
output_device_menu = tk.OptionMenu(root, output_device_var, *output_device_names)
output_device_menu.grid(row=3, column=0, columnspan=2)

broadcast_default_var = tk.BooleanVar()
broadcast_default_checkbox = tk.Checkbutton(root, text="Play through default device", variable=broadcast_default_var)
broadcast_default_checkbox.grid(row=4, column=0, columnspan=2)

broadcast_selected_var = tk.BooleanVar()
broadcast_selected_checkbox = tk.Checkbutton(root, text="Play through selected device", variable=broadcast_selected_var)
broadcast_selected_checkbox.grid(row=5, column=0, columnspan=2)

play_button = tk.Button(root, text="Save Audio", command=process_input)
play_button.grid(row=6, column=0, columnspan=2)

root.mainloop()
