import os
import pandas as pd
import numpy as np
import torch
import torchaudio

path = "/Project/wav/"
all_data = sorted(os.listdir(path))

# divide train, devel, and test data
train_data = all_data[:31745]
devel_data = all_data[31745:41670]
test_data = all_data[41670:]

# Preprocess Audio Data
class AudioProcessing():
    # load audio file
    def load(audio_file):
        signal, sr = torchaudio.load(audio_file)
        signal = torchaudio.functional.resample(signal, orig_freq=sr, new_freq=16000)
        return signal, sr

# set up hubert_large signals with max_Li    
train_signal_hubert_large = np.zeros((31745, 398, 1024))
devel_signal_hubert_large = np.zeros((9925, 398, 1024))
test_signal_hubert_large = np.zeros((10143, 398, 1024))

# set up train, devel emotion
train_emotion = np.zeros((31745,9))
devel_emotion = np.zeros((9925,9))

audio_train_data = pd.read_csv("train.csv")
audio_devel_data = pd.read_csv("devel.csv")
audio_test_data = pd.read_csv("test.csv")

# train signal hubert_large and train emotion
for i in range(len(train_data)):
    audio_file = path + train_data[i]
    signal, sr = AudioProcessing.load(audio_file)
    bundle = torchaudio.pipelines.HUBERT_LARGE
    model_hubert_large = bundle.get_model()
    signal_hubert_large,_ = model_hubert_large(signal)
    signal_array = signal_hubert_large.detach().numpy()
    train_signal_hubert_large[i, :signal_array.shape[1], :signal_array.shape[2]] = signal_array[0]
    # emotion
    row = audio_train_data[audio_train_data["filename"] == str(train_data[i])]
    emotion=[]
    emotion.append(row.loc[i,"Anger"])
    emotion.append(row.loc[i,"Boredom"])
    emotion.append(row.loc[i,"Calmness"])
    emotion.append(row.loc[i,"Concentration"])
    emotion.append(row.loc[i,"Determination"])
    emotion.append(row.loc[i,"Excitement"])
    emotion.append(row.loc[i,"Interest"])
    emotion.append(row.loc[i,"Sadness"])
    emotion.append(row.loc[i,"Tiredness"])
    train_emotion[i] = emotion
    print(f"processing file {i}")

folder_path = '/data/'
os.makedirs(folder_path, exist_ok=True)

# save hubert_large train signal
file_path = os.path.join(folder_path, 'train_signal_hubert_large.npy')
np.save(file_path, train_signal_hubert_large)

# save train emotion
file_path_emotion = os.path.join(folder_path, 'train_emotion.npy')
np.save(file_path_emotion, train_emotion)

# devel signal hubert_large and devel emotion
for i in range(len(devel_data)):
    audio_file = path + devel_data[i]
    signal, sr = AudioProcessing.load(audio_file)
    bundle = torchaudio.pipelines.HUBERT_LARGE
    model_hubert_large = bundle.get_model()
    signal_hubert_large,_ = model_hubert_large(signal)
    signal_array = signal_hubert_large.detach().numpy()
    devel_signal_hubert_large[i, :signal_array.shape[1], :signal_array.shape[2]] = signal_array[0]
    # emotion
    row = audio_devel_data[audio_devel_data["filename"] == str(devel_data[i])]
    emotion=[]
    emotion.append(row.loc[i,"Anger"])
    emotion.append(row.loc[i,"Boredom"])
    emotion.append(row.loc[i,"Calmness"])
    emotion.append(row.loc[i,"Concentration"])
    emotion.append(row.loc[i,"Determination"])
    emotion.append(row.loc[i,"Excitement"])
    emotion.append(row.loc[i,"Interest"])
    emotion.append(row.loc[i,"Sadness"])
    emotion.append(row.loc[i,"Tiredness"])
    devel_emotion[i] = emotion
    print(f"processing file {i}")

# save hubert_large devel signal
file_path = os.path.join(folder_path, 'devel_signal_hubert_large.npy')
np.save(file_path, devel_signal_hubert_large)

# save devel emotion
file_path_emotion = os.path.join(folder_path, 'devel_emotion.npy')
np.save(file_path_emotion, devel_emotion)

# test_signal_hubert
for i in range(len(test_data)):
    audio_file = path + test_data[i]
    signal, sr = AudioProcessing.load(audio_file)
    bundle = torchaudio.pipelines.HUBERT_LARGE
    model_hubert_large = bundle.get_model()
    signal_hubert_large,_ = model_hubert_large(signal)
    signal_array = signal_hubert_large.detach().numpy()
    test_signal_hubert_large[i, :signal_array.shape[1], :signal_array.shape[2]] = signal_array[0]
    print(f"processing file {i}")

# save hubert_large test signal
file_path = os.path.join(folder_path, 'test_signal_hubert_large.npy')
np.save(file_path, test_signal_hubert_large)