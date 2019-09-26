
args_LeNet = {"model_type":"LeNet", "num_epochs":10, "batch_size":100, "learning_rate":0.0012}

args = args_LeNet
batch_size = args["batch_size"]
args['flexible_act'] = True
args["batch_norm"] = False
args["dataset"] = "CIFAR10"
args["reg_act"] = 0
args["patience"] = 50
args["reg_lambda"] = 0.032 
args["reg_dev"] = 0 
args["learning_rate"] = 0.0012

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
    
