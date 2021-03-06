import torch
import torch.nn as nn
import torch.nn.modules.rnn
from enum import IntEnum
import activations

class LSTM(nn.Module):
    def __init__(self, args):

        self.layer_size = args["layer_size"]
        self.num_layers = len(self.layer_size)
        self.num_outputs = args["num_outputs"]
        self.new_acts = args["new_acts"]
        self.layer_type = args["layer_type"]

        super(LSTM, self).__init__()
        lstm_layers = []
        
        for i in range(self.num_layers-1):
            if self.new_acts == True or self.new_acts == "FSig_reg":
                lstm_layers.append(LSTM_cell(self.layer_size[i], self.layer_size[i+1], self.layer_type))
            else:
                lstm_layer = nn.LSTM(self.layer_size[i], self.layer_size[i+1], num_layers = 1, batch_first=True)
                lstm_layers.append(lstm_layer)
            self.lstm_layers = nn.ModuleList(lstm_layers)
            
        self.fc = nn.Linear(self.layer_size[-1], self.num_outputs)
        
    def forward(self, x):

        out = x
        act_para_list = []
        act_para_std_list = []

        for i in range(self.num_layers-1):
            
            h = torch.zeros(1, x.shape[0], self.layer_size[i+1], dtype = torch.float32).to(x.device) 
            c = torch.zeros(1, x.shape[0], self.layer_size[i+1], dtype = torch.float32).to(x.device)
            
            if self.new_acts == True or self.new_acts == "FSig_reg":
                h = h[0,:,:]
                c = c[0,:,:]
            
            lstm_layer = self.lstm_layers[i]
            if self.new_acts == True or self.new_acts == "FSig_reg":
                
                # Use the following three lines instead when use NaiveLSTM_0
                out, _, act_para, act_para_std = lstm_layer(out, (h, c)) #, ind_first = self.ind_first)
                act_para_list.append(act_para)
                act_para_std_list.append(act_para_std)
            else:
                out, _ = lstm_layer(out, (h, c))
       
        out = self.fc(out[:, -1, :])
        
        if self.new_acts == True or self.new_acts == "FSig_reg":
            return out, act_para_list, act_para_std_list
        else:
            return out
    
class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2
    
class LSTM_cell(torch.nn.modules.rnn.RNNBase):
    def __init__(self, input_sz, hidden_sz, layer_type):
        super(NaiveLSTM, self).__init__(mode = "LSTM", input_size = input_sz, hidden_size = hidden_sz)
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        # input gate
        self.W_ii = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i = nn.Parameter(torch.Tensor(hidden_sz))
        # forget gate
        self.W_if = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_f = nn.Parameter(torch.Tensor(hidden_sz))
        # ???
        self.W_ig = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_g = nn.Parameter(torch.Tensor(hidden_sz))
        # output gate
        self.W_io = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_o = nn.Parameter(torch.Tensor(hidden_sz))
         
        self.init_weights()
        
        self.p_act_layer_1 = p_act_layer_sig_tanh(hidden_sz, layer_type) 
        self.p_act_layer_2 = p_act_layer_sig_tanh(hidden_sz, layer_type)
        self.p_act_layer_3 = p_act_layer_sig_tanh(hidden_sz, layer_type)
     
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
         
    def forward(self, x, init_states, ind_first = False, return_type = "seq_tensor"):
    
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = torch.zeros(self.hidden_size).to(x.device), torch.zeros(self.hidden_size).to(x.device)
        else:
            h_t, c_t = init_states
            
        x = x.type(torch.float32)
        h_t = h_t.type(torch.float32)
        c_t = c_t.type(torch.float32)
        
        for t in range(seq_sz): # iterate over the time steps
            x_t = x[:, t, :]
            i_t, p_list_1, ps_list_1 = self.p_act_layer_1(x_t @ self.W_ii + h_t @ self.W_hi + self.b_i)
            f_t, p_list_2, ps_list_2 = self.p_act_layer_2(x_t @ self.W_if + h_t @ self.W_hf + self.b_f)
            g_t = torch.tanh(x_t @ self.W_ig + h_t @ self.W_hg + self.b_g)
            o_t, p_list_3, ps_list_3 = self.p_act_layer_3(x_t @ self.W_io + h_t @ self.W_ho + self.b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(Dim.batch))
        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)

        print("hidden_seq.size()", hidden_seq.size())
        return hidden_seq, (h_t, c_t)

class autoencoder_1(nn.Module):
    def __init__(self, args):
        super(autoencoder_1, self).__init__()
        self.flexible_act = args['flexible_act']
        if self.flexible_act == True or self.flexible_act == "FReLu_reg":
            print("FReLu")
            self.p_act_layer_1 = p_act_layer(input_features = 36, args = args)
            self.p_act_layer_2 = p_act_layer(input_features = 36, args = args)
        elif self.flexible_act == "PReLu":
            print("PReLu")
            self.p_act_layer_1 = nn.PReLU()
            self.p_act_layer_2 = nn.PReLU()
        elif self.flexible_act == "PReLu_reg":
            self.p_act_layer_1 = p_act_layer_prelu(64)
            self.p_act_layer_2 = p_act_layer_prelu(64)
        else:
            print("False")
            self.p_act_layer_1 = nn.ReLU(True)
            self.p_act_layer_2 = nn.ReLU(True)
        
        self.noise = args["noise_level"]
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),            
            nn.Linear(64, 36),
            self.p_act_layer_1, 
            nn.Linear(36, 12), 
            nn.ReLU(True), 
            nn.Linear(12, 6))
        self.decoder = nn.Sequential(
            nn.Linear(6, 12),
            nn.ReLU(True),
            nn.Linear(12, 36),
            self.p_act_layer_2,             
            nn.Linear(36, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Tanh())

    def forward(self, x, x_n = None):
        
        if self.noise == None:
            x = self.encoder(x)
        else:
            x = self.encoder(x_n)
        x = self.decoder(x)
        
        return x
    
class autoencoder_2(nn.Module):
    def __init__(self, args):
        self.flexible_act = args['flexible_act']
        super(autoencoder_2, self).__init__()

        if self.flexible_act == True or self.flexible_act == "FTanh_reg":
            # self.p_act_layer_1 = p_act_layer_all(input_features = 128, layer_type = "tanh_slope")
            # self.p_act_layer_2 = p_act_layer_all(input_features = 64, layer_type = "tanh_slope")
            self.p_act_layer_1 = p_act_layer_all(input_features = 12, layer_type = "tanh_slope")
            self.p_act_layer_2 = p_act_layer_all(input_features = 12, layer_type = "tanh_slope")
            # self.p_act_layer_5 = p_act_layer_all(input_features = 64, layer_type = "tanh_slope")
            # self.p_act_layer_6 = p_act_layer_all(input_features = 128, layer_type = "tanh_slope")
        else:
            # self.p_act_layer_1 = nn.Tanh()
            # self.p_act_layer_1 = nn.Tanh()
            # self.p_act_layer_2 = nn.Tanh()
            self.p_act_layer_1 = nn.Tanh()
            self.p_act_layer_2 = nn.Tanh()
            # self.p_act_layer_5 = nn.Tanh()
            # self.p_act_layer_6 = nn.Tanh()
        
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.Tanh(),
            # self.p_act_layer_1, 
            nn.Linear(128, 64),
            nn.Tanh(),
            # self.p_act_layer_2, # 
            nn.Linear(64, 12),
            # nn.Tanh(),
            self.p_act_layer_1, # 
            nn.Linear(12, 3),   # compress to 3 features which can be visualized in plt
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            # nn.Tanh(),
            self.p_act_layer_2, 
            nn.Linear(12, 36),
            nn.Tanh(),
            nn.Linear(36, 64),
            nn.Tanh(),
            # self.p_act_layer_5, # nn.Tanh(),
            nn.Linear(64, 128),
            # nn.Tanh(),
            # self.p_act_layer_6, # nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        # return encoded, decoded
        return decoded
    
class conv_autoencoder(nn.Module):
    def __init__(self, args):
        super(conv_autoencoder, self).__init__()
        print("conv_autoencoder")
        self.noise = args["noise_level"]
        self.flexible_act = args["flexible_act"]
        
        if self.flexible_act == True or self.flexible_act == "FReLu_reg":
            print("FReLu")
            self.p_act_layer_1 = p_act_layer(input_features = 3, args = args)
            self.p_act_layer_2 = p_act_layer(input_features = 5, args = args)
        elif self.flexible_act == "PReLu":
            print("PReLu")
            self.p_act_layer_1 = nn.PReLU()
            self.p_act_layer_2 = nn.PReLU()
            self.p_act_layer_3 = nn.PReLU()
            self.p_act_layer_4 = nn.PReLU()
        elif self.flexible_act == "PReLu_reg":
            self.p_act_layer_1 = p_act_layer_prelu(12)
            self.p_act_layer_2 = p_act_layer_prelu(12)
            self.p_act_layer_3 = p_act_layer_prelu(12)
            self.p_act_layer_4 = p_act_layer_prelu(12)
        elif self.flexible_act == "Relu_Elu" or self.flexible_act == "Relu_Elu_reg" or self.flexible_act == "Relu_Elu_reg_1":
            self.p_act_layer_1 = p_act_layer_relu_elu(input_features = 3)
            self.p_act_layer_2 = p_act_layer_relu_elu(input_features = 5)
            self.p_act_layer_3 = p_act_layer_relu_elu(input_features = 10)
            self.p_act_layer_4 = p_act_layer_relu_elu(input_features = 15)
        elif self.flexible_act == "Gelu":
            print("Gelu")
            self.p_act_layer_1 = GELU()
            self.p_act_layer_2 = GELU()
            self.p_act_layer_3 = GELU()
            self.p_act_layer_4 = GELU()
        elif self.flexible_act == "Elu":
            print("Elu")
            self.p_act_layer_1 = nn.ELU()
            self.p_act_layer_2 = nn.ELU()
            self.p_act_layer_3 = nn.ELU()
            self.p_act_layer_4 = nn.ELU()
        else:
            print("False")
            self.p_act_layer_1 = nn.ReLU(True)
            self.p_act_layer_2 = nn.ReLU(True)
            self.p_act_layer_3 = nn.ReLU(True)
            self.p_act_layer_4 = nn.ReLU(True)
            
        if args["dataset"]=="CIFAR10":
            dim = 3
        if args["dataset"]=="MNIST" or args["dataset"]=="FMNIST":
            dim = 1
        
        self.encoder = nn.Sequential(
            nn.Conv2d(dim, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            # nn.ReLU(True),
            self.p_act_layer_3,
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            # nn.ReLU(True),
            self.p_act_layer_1,
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            # nn.ReLU(True),
            self.p_act_layer_2,
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            # nn.ReLU(True),
            self.p_act_layer_4,
            nn.ConvTranspose2d(8, dim, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x, x_n = None):
        
        if self.noise == None:
            x = self.encoder(x)
        else:
            x = self.encoder(x_n)
        x = self.decoder(x)
        
        return x
    
class conv_autoencoder_cifar(nn.Module):
    def __init__(self, args):
        super(conv_autoencoder_cifar, self).__init__()
        print("conv_autoencoder_cifar")
        self.flexible_act = args["flexible_act"]
        if self.flexible_act == True or self.flexible_act == "FReLu_reg":
            print("FReLu")
            self.p_act_layer_1 = p_act_layer(input_features = 3, args = args)
            self.p_act_layer_2 = p_act_layer(input_features = 5, args = args)
        elif self.flexible_act == "PReLu":
            print("PReLu")
            self.p_act_layer_1 = nn.PReLU()
            self.p_act_layer_2 = nn.PReLU()
            self.p_act_layer_3 = nn.PReLU()
            self.p_act_layer_4 = nn.PReLU()
            self.p_act_layer_5 = nn.PReLU()
            self.p_act_layer_6 = nn.PReLU()
        elif self.flexible_act == "PReLu_reg":
            self.p_act_layer_1 = p_act_layer_prelu(12)
            self.p_act_layer_2 = p_act_layer_prelu(12)
        elif self.flexible_act == "Relu_Elu" or self.flexible_act == "Relu_Elu_reg" or self.flexible_act == "Relu_Elu_reg_1":
            self.p_act_layer_1 = p_act_layer_relu_elu(input_features = 16)
            self.p_act_layer_2 = p_act_layer_relu_elu(input_features = 8)
            self.p_act_layer_3 = p_act_layer_relu_elu(input_features = 4)
            self.p_act_layer_4 = p_act_layer_relu_elu(input_features = 8)
            self.p_act_layer_5 = p_act_layer_relu_elu(input_features = 16)
            self.p_act_layer_6 = p_act_layer_relu_elu(input_features = 32)
        elif self.flexible_act == "Gelu":
            print("Gelu")
            self.p_act_layer_1 = GELU()
            self.p_act_layer_2 = GELU()
            self.p_act_layer_3 = GELU()
            self.p_act_layer_4 = GELU()
            self.p_act_layer_5 = GELU()
            self.p_act_layer_6 = GELU()
        elif self.flexible_act == "Elu":
            print("Elu")
            self.p_act_layer_1 = nn.ELU()
            self.p_act_layer_2 = nn.ELU()
            self.p_act_layer_3 = nn.ELU()
            self.p_act_layer_4 = nn.ELU()
            self.p_act_layer_5 = nn.ELU()
            self.p_act_layer_6 = nn.ELU()
        else:
            print("False")
            self.p_act_layer_1 = nn.ReLU(True)
            self.p_act_layer_2 = nn.ReLU(True)
            self.p_act_layer_3 = nn.ReLU(True)
            self.p_act_layer_4 = nn.ReLU(True)
            self.p_act_layer_5 = nn.ReLU(True)
            self.p_act_layer_6 = nn.ReLU(True)
            
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            # nn.ReLU(),
            self.p_act_layer_1,
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            # nn.ReLU(),
            self.p_act_layer_2,
            nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            self.p_act_layer_3,
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            self.p_act_layer_4,
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            self.p_act_layer_5,
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            self.p_act_layer_6,
            # nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    
class LeNet(nn.Module):
    def __init__(self, args):
        super(LeNet, self).__init__()
        self.new_acts = args["flexible_act"]
        if args["dataset"] == "MNIST":
            self.conv1 = nn.Conv2d(1, 6, 5)
        if args["dataset"] == "CIFAR10":
            self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        self.bn_layers_1 = nn.BatchNorm1d(120)
        self.bn_layers_2 = nn.BatchNorm1d(84)
        
        print("self.new_acts:", self.new_acts)
        if self.new_acts == False:
            print("False")
            self.p_act_layer_1 = nn.ReLU()
            self.p_act_layer_2 = nn.ReLU()
        elif self.new_acts == "PReLu_reg":
            print("PReLu_reg")
            self.p_act_layer_1 = nn.ReLU()
            self.p_act_layer_2 = p_act_layer_prelu(84) #  nn.PReLU() #
        elif self.new_acts == "PReLu":
            print("PReLu")
            self.p_act_layer_1 = nn.ReLU()
            self.p_act_layer_2 = nn.PReLU() #  nn.PReLU() #           
        else:
            print("new_act!")
            self.p_act_layer_1 = nn.ReLU()
            # self.p_act_layer_1 = p_act_layer(120, args)
            self.p_act_layer_2 = p_act_layer(84, args)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        # out = F.relu(self.fc1(out))
        # out = F.relu(self.fc2(out))
        out = self.p_act_layer_1(self.bn_layers_1(self.fc1(out)))
        out = self.p_act_layer_2(self.bn_layers_2(self.fc2(out)))
    
        out = self.fc3(out)
        return out
