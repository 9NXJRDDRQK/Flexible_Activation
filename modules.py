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
                # print(len(lstm_layer(out, (h, c))))
                # print("new_acts!!!")
                # print("out.size()", out.size())
                
                # out, _ = lstm_layer(out, (h, c)) #, ind_first = self.ind_first)
                # h, c = _
                
                # Use the following three lines instead when use NaiveLSTM_0
                out, _, act_para, act_para_std = lstm_layer(out, (h, c)) #, ind_first = self.ind_first)
                act_para_list.append(act_para)
                act_para_std_list.append(act_para_std)
            else:
                out, _ = lstm_layer(out, (h, c))
        
        # print("out.size: before fc", out.size())
        # print("out[:, -1, :]", out[:, -1, :].size())
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
        
        self.p_act_layer_1 = p_act_layer_all(hidden_sz, layer_type) 
        self.p_act_layer_2 = p_act_layer_all(hidden_sz, layer_type)
        self.p_act_layer_3 = p_act_layer_all(hidden_sz, layer_type)
     
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
         
    def forward(self, x, init_states, ind_first = False, return_type = "seq_tensor"):
    
        # Assumes x is of shape (batch, sequence, feature)
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = torch.zeros(self.hidden_size).to(x.device), torch.zeros(self.hidden_size).to(x.device)
        else:
            h_t, c_t = init_states
            
        x = x.type(torch.float32)
        # print("x.size():after", x.size())
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
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        # hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        print("hidden_seq.size()", hidden_seq.size())
        return hidden_seq, (h_t, c_t)
