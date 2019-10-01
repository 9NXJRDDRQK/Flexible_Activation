import train
import modules
import activations

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.is_available()

# Multi_TS: G20 Stock Indices

BIST = pd.read_csv("Datasets/00_Benchmarks/G20_Indices/BIST 100 Historical Data (Turkey).csv")
Bovespa = pd.read_csv("Datasets/00_Benchmarks/G20_Indices/Bovespa Historical Data (Brazil).csv")
CAC = pd.read_csv("Datasets/00_Benchmarks/G20_Indices/CAC 40 Historical Data (France).csv")
DAX = pd.read_csv("Datasets/00_Benchmarks/G20_Indices/DAX Historical Data (German).csv")
FTSE_100 = pd.read_csv("Datasets/00_Benchmarks/G20_Indices/FTSE 100 Historical Data (British).csv")
FTSE_China_A50 = pd.read_csv("Datasets/00_Benchmarks/G20_Indices/FTSE China A50 Historical Data (China).csv")
FTSE_MIB = pd.read_csv("Datasets/00_Benchmarks/G20_Indices/FTSE Italia Mid Cap Historical Data (Italia).csv")
FTSE_JSE = pd.read_csv("Datasets/00_Benchmarks/G20_Indices/FTSE_JSE Top 40 Historical Data (South Africa).csv")
GSPC = pd.read_csv("Datasets/00_Benchmarks/G20_Indices/GSPC (US).csv")
Euro = pd.read_csv("Datasets/00_Benchmarks/G20_Indices/Investing.com Euro Index Historical Data (Euro).csv")
Jakarta = pd.read_csv("Datasets/00_Benchmarks/G20_Indices/Jakarta Stock Exchange Composite Index Historical Data (Indonesia).csv")
KOSPI_50 = pd.read_csv("Datasets/00_Benchmarks/G20_Indices/KOSPI 50 Historical Data (Korea).csv")
Merval = pd.read_csv("Datasets/00_Benchmarks/G20_Indices/Merval Historical Data (Agentina).csv")
Nifty = pd.read_csv("Datasets/00_Benchmarks/G20_Indices/Nifty 50 Historical Data (India).csv")
Nikkei = pd.read_csv("Datasets/00_Benchmarks/G20_Indices/Nikkei 225 Historical Data (Japan).csv")
RTSI = pd.read_csv("Datasets/00_Benchmarks/G20_Indices/RTSI Historical Data (Russia).csv")
SP_ASX = pd.read_csv("Datasets/00_Benchmarks/G20_Indices/S&P_ASX 200 Historical Data (Australia).csv")
SP_BMV = pd.read_csv("Datasets/00_Benchmarks/G20_Indices/S&P_BMV IPC Historical Data (Mexico).csv")
SP_TSX = pd.read_csv("Datasets/00_Benchmarks/G20_Indices/S&P_TSX Composite Historical Data (Canada).csv")
Tadawul = pd.read_csv("Datasets/00_Benchmarks/G20_Indices/Tadawul All Share Historical Data (Saudi).csv")

url_list = ["Datasets/00_Benchmarks/G20_Indices/S&P_ASX 200 Historical Data (Australia).csv",
           "Datasets/00_Benchmarks/G20_Indices/FTSE 100 Historical Data (British).csv",
            # "Datasets/00_Benchmarks/G20_Indices/GSPC (US).csv",
            "Datasets/00_Benchmarks/G20_Indices/KOSPI 50 Historical Data (Korea).csv",
            "Datasets/00_Benchmarks/G20_Indices/RTSI Historical Data (Russia).csv"
           ]

df = get_mul_var_data(url_list)
df.fillna(0, inplace=True)
data = np.array(df)

print("data.shape", data.shape)

examples = 30
y_examples = 1

args = {"model_code":"DL_TS_LSTM","examples":examples, "y_examples":y_examples, "layer_size":[4,8], 
        "num_outputs": 1, "reg1": 0.0, "reg2": 0.0, "act": "linear", "drop": 0.5, "epochs": 30, "reg_lam":0,
      "batch_size":50, "loss":"mse", "optimizer": "SGD", "learning_rate": 0.01, "rho":0.9, "momentum": 0.9, 
        "beta_1":0.9, "beta_2":0.999, "decay":0.0, "schedule_decay":0.004, "early_stopping":False}

args["dataset"] = "mul_stocks"
args["patience"] = 7
args["new_acts"] = False

args["num_outputs"] = 1
args["reg_class"] = "reg"
args["layer_size"] = [5, 16]
input_mat, target_mat = data_process_TS(data, examples, y_examples)
    
input_mat_train, input_mat_test =  split(input_mat, 0.8)
target_mat_train, target_mat_test =  split(target_mat, 0.8)

input_mat_train, input_mat_val =  split(input_mat_train, 0.8)
target_mat_train, target_mat_val =  split(target_mat_train, 0.8)

train_X, train_Y, val_X, val_Y, test_X, test_Y = input_mat_train, target_mat_train, input_mat_val, target_mat_val, input_mat_test, target_mat_test

args["learning rate"] = 0.019389 
args["reg_lam"] =  0.0250 
args["patience"] = 50
args["epochs"] =  30
args["layer_size"] = [4, 16]
args["layer_type"] = "sigmoid_slope"

eva_list_1 = []
eva_list_2 = []
eva_list_3 = []

val_losses_table_1 = []
val_losses_table_2 = []
val_losses_table_3 = []

act_para_list_list_list = []
act_para_std_list_list_list = []

for i in range(50):
    
    args["new_acts"] = True
    eva_2, model_2, avg_train_losses_2, avg_valid_losses_2, time_list_2, act_para_list_list, act_para_std_list_list = model_train_rnn(train_X, train_Y, val_X, val_Y, test_X, test_Y, HP = args)
    try:
        eva_2 = eva_2.item()
    except:
        eva_2 = eva_2
    print("eva_2:", eva_2)
    eva_list_2.append(eva_2)
    val_losses_table_2.append(avg_valid_losses_2)
    
    args["new_acts"] = False
    eva_1, model_1, avg_train_losses_1, avg_valid_losses_1, time_list_1 = model_train_rnn(train_X, train_Y, val_X, val_Y, test_X, test_Y, HP = args)
    try:
        eva_1 = eva_1.item()
    except:
        eva_1 = eva_1
    print("eva_1:", eva_1)
    eva_list_1.append(eva_1)
    val_losses_table_1.append(avg_valid_losses_1)
    
    args["new_acts"] = "FSig_reg"
    eva_3, model_3, avg_train_losses_3, avg_valid_losses_3, time_list_3, act_para_list_list, act_para_std_list_list = model_train_rnn(train_X, train_Y, val_X, val_Y, test_X, test_Y, HP = args)
    try:
        eva_3 = eva_3.item()
    except:
        eva_3 = eva_3
    print("eva_3:", eva_3)
    eva_list_3.append(eva_3)
    val_losses_table_3.append(avg_valid_losses_3)
    
