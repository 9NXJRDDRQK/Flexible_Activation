
args_LeNet = {"model_type":"LeNet", "num_epochs":10, "batch_size":100, "learning_rate":0.001}

args = args_LeNet
batch_size = args["batch_size"]
args['flexible_act'] = True
args["batch_norm"] = False
args["dataset"] = "CIFAR10"
args["reg_act"] = 0
# args["reg_lambda"] = 0.01
args["patience"] = 50
args["reg_lambda"] = 0.032 # 0.04123 # 0.0004208 # 0.0036946 # 0.0429184 # 0 # 0.209566 # 0.0698 # 0.02596 # 0.0479 # 0.2
args["reg_dev"] = 0 # 0.1095 # 0.34437 # 0.02377 # 0.003772 # 0.1645
args["learning_rate"] = 0.001

# CIFAR10: "{'reg_lambda': 0.04123046021024116, 'reg_dev': 0.3443737191272257}",65.15
# "{'reg_lambda': 0.007639503921799092, 'reg_dev': 0.10952417785146537}",65.25
# MNIST "{'reg_lambda': 0.0009625171801829108, 'reg_dev': 0.09193904789387754}",99.16

# "{'reg_lambda': 0.0004208101021752669, 'reg_dev': 0.023776613767154512}",51.83
# "{'reg_lambda': 0.06979813907830665, 'reg_dev': 0.16451905877536627}",49.33
# "{'reg_lambda': 0.20956623994804346, 'reg_dev': 0.003772042493416998}",49.27

# Image preprocessing modules with torchvision.transforms
if args["model_type"] == "ResNet":
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()])
else:
    transform=transforms.ToTensor()
    
if args["dataset"] == "MNIST":  
    transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
if args["dataset"] == "MNIST":
    train_dataset = torchvision.datasets.MNIST(root='Datasets', 
                                           train=True, 
                                           transform=transform,  
                                           download=True)
    test_dataset = torchvision.datasets.MNIST(root='Datasets', 
                                          train=False, 
                                          transform=transform)
    
if args["dataset"] == "CIFAR10":
    train_dataset = torchvision.datasets.CIFAR10(root='Datasets', 
                                           train=True, 
                                           transform=transform,  
                                           download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='Datasets', 
                                          train=False, 
                                          transform=transform)
    
# print("len(test_dataset)", len(test_dataset))


train_lost_len = int(0*len(train_dataset))
train_len = len(train_dataset) - train_lost_len

eva_list_0 = []
eva_list_1 = []
eva_list_2 = []
# eva_new_list = []
val_losses_table_0 = []
val_losses_table_1 = []
val_losses_table_2 = []

args['rand_act'] = True

for i in range(100):
    
    train_dataset_1, train_lost = torch.utils.data.random_split(train_dataset, lengths=[train_len, train_lost_len])

    train_len_1 = int(0.8*len(train_dataset_1))
    valid_len = len(train_dataset_1) - train_len_1
    train_dataset_2, val_dataset = torch.utils.data.random_split(train_dataset_1, lengths=[train_len_1, valid_len])
    # print("train_len", train_len, "valid_len", valid_len)                                    

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset_2, 
                                           batch_size=batch_size, 
                                           shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)
    data = [train_loader, val_loader, test_loader] 
    
    args['flexible_act'] = "PReLu"
    eva_1, model, train_losses_1, valid_losses_1, time_list_1 = model_train(data, args = args)
    eva_list_1.append(eva_1)
    val_losses_table_1.append(valid_losses_1)
    args['flexible_act'] = "PReLu_reg"
    eva_2, model, train_losses_2, valid_losses_2, time_list_2 = model_train(data, args = args)
    eva_list_2.append(eva_2)
    val_losses_table_2.append(valid_losses_2)
    args['flexible_act'] = False
    eva_0, model, train_losses_0, valid_losses_0, time_list_0 = model_train(data, args = args)
    eva_list_0.append(eva_0)
    val_losses_table_0.append(valid_losses_0)
    
    
    """
    args['flexible_act'] = True
    eva_new, model, train_losses_new, valid_losses_new, time_list_new = model_train(data, args = args)
    eva_new_list.append(eva_new)
    val_losses_table_new.append(valid_losses_new)
    """
    
# wd = os.getcwd()
datetime = strftime("%Y-%m-%d %H:%M:%S",gmtime())
fn =  datetime + "_f_act_" + args["dataset"]

if args["batch_norm"] == True:
    fn = fn + "_batch_norm"
    
# rows = zip(update_list, eva_list)
# rows = zip(eva_list)eva_list_1
rows = zip(eva_list_0, eva_list_1, eva_list_2)
         
val_losses_list_0 = []
val_losses_list_1 = []
val_losses_list_2 = []

for i in range(len(val_losses_table_0)):
    val_losses_list_0 = val_losses_list_0 + val_losses_table_0[i]
    val_losses_list_1 = val_losses_list_1 + val_losses_table_1[i]
    val_losses_list_2 = val_losses_list_2 + val_losses_table_2[i]
# print(val_losses_list)

rows_1 = zip(val_losses_list_0, val_losses_list_1, val_losses_list_2)

with open(fn, "w") as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)
    for row in rows_1:
        writer.writerow(row)
