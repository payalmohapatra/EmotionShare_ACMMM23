import os
import torch
import torchaudio

path = "/Project/wav/"
all_data = sorted(os.listdir(path))

# Preprocess Audio Data
class AudioProcessing():
    # load audio file
    def load(audio_file):
        signal, sr = torchaudio.load(audio_file)
        signal = torchaudio.functional.resample(signal, orig_freq=sr, new_freq=16000)
        return signal, sr

# define maximum sequence length
max_Li = []
for i in range(len(all_data)):
    audio_file = path + all_data[i]
    signal, sr = AudioProcessing.load(audio_file)
    bundle = torchaudio.pipelines.HUBERT_LARGE
    model_hubert_large = bundle.get_model()
    signal_hubert_large,_ = model_hubert_large(signal)
    max_Li.append(signal_hubert_large.shape[1])
    print(f"file {i} is processing, the max_Li is {signal_hubert_large.shape[1]}")
