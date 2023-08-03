import os
import torch
import torchaudio
import numpy as np

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
Li_all = []
for i in range(len(all_data)):
    audio_file = path + all_data[i]
    signal, sr = AudioProcessing.load(audio_file)
    bundle = torchaudio.pipelines.HUBERT_LARGE
    model_hubert_large = bundle.get_model()
    signal_hubert_large,_ = model_hubert_large(signal)
    Li_all.append(signal_hubert_large.shape[1])
    print(f"file {i} is processing, the max_Li is {signal_hubert_large.shape[1]}")

max_Li = max(Li_all)

Li_all = np.array(Li_all)
np.save('Li_all.npy', Li_all)
