import os
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn 
from torch.autograd import Variable
import numpy as np
import math
from helper_functions import set_seed
from torch.utils.tensorboard import SummaryWriter
import scipy.stats as stats

folder_path = '/data/'
file_train_hubert = 'train_signal_hubert_large.npy'
file_train_emotion = 'train_emotion.npy'
file_devel_hubert = 'devel_signal_hubert_large.npy'
file_devel_emotion = 'devel_emotion.npy'

# train signal and emotion
train_path_hubert = os.path.join(folder_path, file_train_hubert)
train_path_emotion = os.path.join(folder_path, file_train_emotion)
train_signal_hubert = np.load(train_path_hubert)
train_emotion = np.load(train_path_emotion)
train_signal_hubert_transpose = np.transpose(train_signal_hubert, (0, 2, 1))

# devel signal and emotion
devel_path_hubert = os.path.join(folder_path, file_devel_hubert)
devel_path_emotion = os.path.join(folder_path, file_devel_emotion)
devel_signal_hubert = np.load(devel_path_hubert)
devel_emotion = np.load(devel_path_emotion)
devel_signal_hubert_transpose = np.transpose(devel_signal_hubert, (0, 2, 1))

# load li_all
Li_all = np.load('Li_all.npy')
Li_all_list = Li_all.tolist()

# devide train, devel
Li_train = Li_all_list[:31745]
Li_devel = Li_all_list[31745:41670]

set_seed(2711)
writer = SummaryWriter(comment = 'lr0.0001tirednessmeanvaluehubertlarge')

# get real length of each audio
Li_train_divided = np.floor(np.array(Li_train) / 2)
Li_devel_divided = np.floor(np.array(Li_devel) / 2)
a = Li_train_divided.astype(int)
b = Li_devel_divided.astype(int)

class TrainDataset(Dataset):
    
    def __init__(self, train_signal_hubert_transpose, train_emotion, a):
        self.train_signal_hubert_transpose = torch.from_numpy(train_signal_hubert_transpose.astype(np.float32))
        self.train_emotion = torch.from_numpy(train_emotion.astype(np.float32))
        self.n_samples = train_signal_hubert_transpose.shape[0]
        self.a = torch.from_numpy(a)
    
    def __getitem__(self,index):
        return self.train_signal_hubert_transpose[index], self.train_emotion[index], self.a[index]
    
    def __len__(self):
        return self.n_samples
    
class DevelDataset(Dataset):
    
    def __init__(self, devel_signal_hubert_transpose, devel_emotion, b):
        self.devel_signal_hubert_transpose = torch.from_numpy(devel_signal_hubert_transpose.astype(np.float32))
        self.devel_emotion = torch.from_numpy(devel_emotion.astype(np.float32))
        self.n_samples = devel_signal_hubert_transpose.shape[0]
        self.b = torch.from_numpy(b)
    
    def __getitem__(self,index):
        return self.devel_signal_hubert_transpose[index], self.devel_emotion[index], self.b[index]
    
    def __len__(self):
        return self.n_samples

train_dataset = TrainDataset(train_signal_hubert_transpose, train_emotion, a)
devel_dataset = DevelDataset(devel_signal_hubert_transpose, devel_emotion, b)

train_loader = DataLoader(dataset = train_dataset,
                          batch_size = 128,
                          shuffle = True,
                          num_workers = 2)

devel_loader = DataLoader(dataset = devel_dataset,
                          batch_size = 128,
                          shuffle = False,
                          num_workers = 2)

input_size = 192
sequence_length = 199
hidden_size = 512
num_layers = 2
num_classes = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
class CNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(CNNLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.layer_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1024,out_channels=768,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=768,out_channels=384,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=384,out_channels=192,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2)
        )
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_emotion_8 = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, num_classes),
            nn.ReLU()
        )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self,x, a):
        cnn_out = self.layer_cnn(x)
        lstm_input = cnn_out.permute(0,2,1)
        pack = nn.utils.rnn.pack_padded_sequence(lstm_input, a, batch_first = True, enforce_sorted = False)
        h0 = Variable(torch.zeros(self.num_layers, lstm_input.size(0), self.hidden_size).to(device))
        c0 = Variable(torch.zeros(self.num_layers, lstm_input.size(0), self.hidden_size).to(device))
        lstm_out,_ = self.lstm(pack, (h0,c0))
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        this_batch_len = unpacked.size(1)
        lstm_out = unpacked
        fc_emotion_8_out = torch.zeros((x.shape[0], lstm_out.shape[2])).to(device)
        for i in range(x.shape[0]):
            fc_emotion_8_out[i,:] = torch.mean(lstm_out[i,0:a[i],:],dim=0)
        fc_emotion_8_out = self.fc_emotion_8(fc_emotion_8_out)
        return fc_emotion_8_out
    
cnnlstm_model = CNNLSTM(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(cnnlstm_model.parameters(), lr = 0.0001)
train_steps = len(train_loader)
devel_steps = len(devel_loader)

def train(epoch):
    cnnlstm_model.train()
    avg_loss = 0.0
    for i, (signals, emotions, a) in enumerate(train_loader):
        signals = signals.to(device)
        emotion_8 = emotions[:,8].to(device).reshape(-1,1)
        lstm_output_8 = cnnlstm_model(signals, a)
        loss_8 = criterion(lstm_output_8, emotion_8)
        avg_loss = (avg_loss * i + loss_8.item())/(i+1)
        optimizer.zero_grad()
        loss_8.backward()
        optimizer.step()
        if (i+1) % 1 == 0:
            print (f'Epoch [{epoch+1}], Step [{i+1}/{train_steps}], avg_Loss: {avg_loss:.4f}, Loss = {loss_8:.4f}')
    writer.add_scalar("0.0001_loss_8_tiredness_meanvalue_hubert_large/train", avg_loss, epoch)

def evaluate(epoch):
    cnnlstm_model.eval()
    act = []
    pre = []
    with torch.no_grad():
        act = torch.zeros((100000,1)).to(device)
        pre = torch.zeros((100000,1)).to(device)
        count_valid = 0
        for i, (signals, emotions, b) in enumerate(devel_loader):
            signals = signals.to(device)
            emotion_8 = emotions[:,8].to(device).reshape(-1,1)
            lstm_output_8 = cnnlstm_model(signals, b)
            size = len(emotion_8)
            act[count_valid:count_valid+size,:] = emotion_8
            pre[count_valid:count_valid+size,:] = lstm_output_8
            count_valid += size
            if (i+1) % 1 == 0:
                print (f'eval_Epoch [{epoch+1}], eval_Step [{i+1}/{devel_steps}]')
        act = act[0:count_valid,:].cpu().numpy()
        pre = pre[0:count_valid,:].cpu().numpy()
        act = np.reshape(act,(-1,1))
        pre = np.reshape(pre,(-1,1))
        correlation_8 = stats.spearmanr(act, pre)[0]
        writer.add_scalar("0.0001_eval_correlation_8_tiredness_meanvalue_hubert_large/devel", correlation_8, epoch)

def save_checkpoint(state, filename='/data/tirednesshubertlarge/checkpoint.pth.tar'):
    torch.save(state, filename)

iters = 150
for epoch in range(iters):
    train(epoch)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': cnnlstm_model.state_dict(),
        'optimizer' : optimizer.state_dict()
    }, filename=f'/data/tirednesshubertlarge/epoch_{epoch+1}.pth.tar')
    evaluate(epoch)