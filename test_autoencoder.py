if not os.path.exists('Results/AE_DAE_VAE/mlp_img'):
    os.mkdir('Results/AE_DAE_VAE/mlp_img')

args = {"dataset": "MNIST", "model_type":"autoencoder_1", "batch_size":128, "num_epochs":50, "learning_rate":1e-3,
       "noise_level": None}
batch_size = args["batch_size"]
args["patience"] = 7
args["reg_lambda"] = 0.901 # 0.02596 # 0.0479 # 0.2 
# Relu: "{'reg_lambda': 0.010940547072057435, 'reg_dev': 0.1282649830528061}",0.03659737482666969
# Tanh: "{'reg_lambda': 0.01355601785329369, 'reg_dev': 0.011969557023590434}",0.03833945468068123

args["reg_dev"] = 0
args['rand_act'] = False

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='Datasets',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=True)

test_dataset = torchvision.datasets.MNIST(root='Datasets', 
                                          train=False, 
                                          transform=transforms.ToTensor())



train_lost_len = int(0.9*len(train_dataset))
train_len = len(train_dataset) - train_lost_len


train_dataset, train_lost = torch.utils.data.random_split(train_dataset, lengths=[train_len, train_lost_len])


train_len = int(0.8*len(train_dataset))
valid_len = len(train_dataset) - train_len
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, lengths=[train_len, valid_len])

# Data loader
train_loader = torch.utils.data.DataLoader(dataset= train_dataset,
                                          batch_size=batch_size, 
                                          shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset= val_dataset,
                                          batch_size=batch_size, 
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset= test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=True)

model = train_autoencoder
data = [train_loader, val_loader, test_loader]


args["learning_rate"] = 0.006457
args["reg_lambda"] = 0.01094 # 0.9014776314524923
args["reg_dev"] = 0.001300511252173409
args["patience"] = 50

eva_list_1 = []
val_losses_table_1 = []

eva_list_2 = []
val_losses_table_2 = []

eva_list_3 = []
val_losses_table_3 = []

for i in range(50):
    
    args['flexible_act'] = False
    eva_1, model_1, avg_train_losses_1, avg_valid_losses_1, time_list = train_autoencoder(data, args)
    try:
        eva_1 = eva_1.item()
    except:
        eva_1 = eva_1
    print("eva_1:", eva_1)
    eva_list_1.append(eva_1)
    val_losses_table_1.append(avg_valid_losses_1)
    
    args['flexible_act'] = True
    eva_2, model_2, avg_train_losses_2, avg_valid_losses_2, time_list = train_autoencoder(data, args)
    try:
        eva_2 = eva_2.item()
    except:
        eva_2 = eva_2
    print("eva_2:", eva_2)
    eva_list_2.append(eva_2)
    val_losses_table_2.append(avg_valid_losses_2)
    
    args['flexible_act'] = "PReLu"
    eva_3, model_3, avg_train_losses_3, avg_valid_losses_3, time_list = train_autoencoder(data, args)
    try:
        eva_3 = eva_3.item()
    except:
        eva_3 = eva_3
    print("eva_3:", eva_3)
    eva_list_3.append(eva_3)
    val_losses_table_3.append(avg_valid_losses_3)
    
path = "Results/Differentiable_Activations/AE/" 
datetime = strftime("%Y-%m-%d %H:%M:%S",gmtime())
fn = path + datetime + "_f_act_" + args["model_type"]
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
