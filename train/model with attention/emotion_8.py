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

set_seed(999)
writer = SummaryWriter(comment = 'lr0.0001tirednessmeanvaluehubertlargeattention')

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
    def __init__(self, input_size, hidden_size, num_layers, num_classes,sequence_length):
        super(CNNLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.layer_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1024,out_channels=768,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=768,out_channels=384,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=384,out_channels=input_size,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2)
        )
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.Q_scale = nn.Sequential(
            nn.Linear(2*hidden_size, hidden_size)
        )
        self.fc_emotion_01 = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU())
        self.fc_emotion_02 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU())
        self.fc_emotion_03 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU())
        self.fc_emotion_04 = nn.Sequential(
            nn.Linear(64, 1),
            nn.ReLU())
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.ln1 = nn.LayerNorm((sequence_length, input_size))
        self.ln2 = nn.LayerNorm((sequence_length, hidden_size))
        self.ln3 = nn.LayerNorm((1, sequence_length))
        self.ln4 = nn.LayerNorm((256))
        self.ln5 = nn.LayerNorm((128))
        self.ln6 = nn.LayerNorm((64))
    def forward(self,x, a):
        cnn_out = self.layer_cnn(x)
        lstm_input = cnn_out.permute(0,2,1)
        lstm_input = self.ln1(lstm_input)
        pack = nn.utils.rnn.pack_padded_sequence(lstm_input, a, batch_first = True, enforce_sorted = False)
        h0 = Variable(torch.zeros(self.num_layers, lstm_input.size(0), self.hidden_size).to(device))
        c0 = Variable(torch.zeros(self.num_layers, lstm_input.size(0), self.hidden_size).to(device))
        lstm_out,hidden_units = self.lstm(pack, (h0,c0))
        hidden_units = hidden_units[0]
        hidden_units = hidden_units.permute(1,0,2)
        hidden_units = nn.Flatten()(hidden_units)
        hidden_units = self.Q_scale(hidden_units)
        hidden_units = hidden_units.reshape((hidden_units.shape[0],1, -1))
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        this_batch_len = unpacked.size(1)
        lstm_out = unpacked
        Q = hidden_units
        K = lstm_out
        V = lstm_out
        QK = torch.matmul(Q, K.permute(0,2,1))/(K.shape[2])**0.5
        QK_d = torch.zeros((QK.shape[0], QK.shape[1], lstm_input.shape[1])).to(device)
        for i in range(QK.shape[0]):
            QK_d[i,:,0:int(a[i])] = nn.Softmax(dim=1)(QK[i,:,0:int(a[i])])
        QK = QK_d
        QK = self.ln3(QK)
        QK = QK[:,:,0:V.shape[1]]
        A = torch.matmul(QK,V)
        lstm_attn_out = A[:,0,:]
        emo = self.fc_emotion_01(lstm_attn_out)
        lstm_attn_out = self.ln4(emo)
        emo = self.fc_emotion_02(lstm_attn_out)
        lstm_attn_out = self.ln5(emo)
        emo = self.fc_emotion_03(lstm_attn_out)
        lstm_attn_out = self.ln6(emo)
        emo = self.fc_emotion_04(lstm_attn_out)
        return emo
    
cnnlstm_model = CNNLSTM(input_size, hidden_size, num_layers, num_classes, sequence_length).to(device)
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
    writer.add_scalar("0.0001_loss_8_tiredness_meanvalue_hubert_large_attention/train", avg_loss, epoch)

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
        writer.add_scalar("0.0001_eval_correlation_8_tiredness_meanvalue_hubert_large_attention/devel", correlation_8, epoch)

def save_checkpoint(state, filename='/data/tirednesshubertlargeattentionseed/checkpoint.pth.tar'):
    torch.save(state, filename)

iters = 150
for epoch in range(iters):
    train(epoch)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': cnnlstm_model.state_dict(),
        'optimizer' : optimizer.state_dict()
    }, filename=f'/data/tirednesshubertlargeattentionseed/epoch_{epoch+1}.pth.tar')
    evaluate(epoch)