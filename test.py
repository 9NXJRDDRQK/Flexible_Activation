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


args["learning rate"] = 0.019389 # [16, 16] 0.004322 # [16, 8] 0.019389 # [8,4,4] 0.003477 # [16] 0.00671 # 0.001419 # 0.01 #0.00135
args["reg_lam"] =  0.0250 # 0.0250 # [16, 16] 0.00215766 # [16, 8] 0.00136  # [8, 4 ,4]0.0636 # [16] 0.042959
args["patience"] = 50
args["epochs"] =  30
args["layer_size"] = [4, 16]
# args["seed"] = 147

# reg_lam : [16] 0.03192, 0.0544, 0.7583
# lr: [16, 16] 0.004322 # [16, 8] 0.019389 # [8,4,4] 0.003477 # [16] 0.00671

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
    

datetime = strftime("%Y-%m-%d %H:%M:%S",gmtime())
fn = datetime + "_f_act_lstm_sigmoid_ind_1" #+ str(val)
# rows = zip(update_list, eva_list)
# rows = zip(eva_list)
rows = zip(eva_list_1, eva_list_2, eva_list_3)

val_losses_list_1 = []
val_losses_list_2 = []
val_losses_list_3 = []

for i in range(len(val_losses_table_1)):
    val_losses_list_1 = val_losses_list_1 + val_losses_table_1[i]
    val_losses_list_2 = val_losses_list_2 + val_losses_table_2[i]
    val_losses_list_3 = val_losses_list_3 + val_losses_table_3[i]
# print(val_losses_list)

rows_1 = zip(val_losses_list_1, val_losses_list_2, val_losses_list_3)

with open(fn, "w") as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)
    for row in rows_1:
        writer.writerow(row)
