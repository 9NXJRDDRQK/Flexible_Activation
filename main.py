def model_train_rnn(X_train, Y_train, X_val, Y_val, X_test, Y_test, args):

    learning_rate = args["learning_rate"]
    num_epochs = args["epochs"]
    batch_size = args["batch_size"]
    reg_lambda= args["reg_lam"]
    reg_class = args["reg_class"]
    patience = args["patience"]
    reg_model = 0 # args["reg_model"]
    
    args_1 = copy.deepcopy(args)
    model = LSTM(args_1).to(device)
    seed = random.randint(1,1000)
    update = False

    # Loss and optimizer
    if reg_class == "class":
        criterion = nn.CrossEntropyLoss()
    if reg_class == "reg":
        criterion = nn.MSELoss()

    optimizer = Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = [] 
    time_list = []
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    t0 = time.clock()
    time_list.append(t0)
    
    minibatches_test = random_mini_batches(X_test, Y_test, batch_size, seed)
    
    # Train the model
    act_para_list_list = []
    act_para_std_list_list = []
    
    for epoch in range(num_epochs):       
        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        minibatches = random_mini_batches(X_train, Y_train, batch_size, seed)
        minibatches_val = random_mini_batches(X_val, Y_val, batch_size, seed)
        total_step = len(minibatches)
        
        i = 0
        for minibatch in minibatches:

            (mX, mY) = minibatch
            mX = torch.tensor(mX).type(torch.FloatTensor)
            if reg_class == "reg":
                mY = torch.tensor(mY).type(torch.FloatTensor)
            if reg_class == "class":
                mY = torch.tensor(mY[:,0]).type(torch.LongTensor)
                
            mX = mX.to(device)
            mY = mY.to(device)
            
            # Forward pass
            if args["new_acts"] == True or args["new_acts"] == "FSig_reg":
                # print("new_acts!!!")
                output, act_para_list, act_para_std_list = model(mX)
                act_para_list_list.append(act_para_list)
                act_para_std_list_list.append(act_para_std_list)
            else:
                output = model(mX)
                
            #try:
            #    output = output.type(torch.FloatTensor).to(device)
            #except: 
            #    output = output[0].type(torch.FloatTensor).to(device)
            # output_final = output_list[-1]
            # print("output_final, mY", output, mY)
            # print("output", output[0], "mY", mY[0], "output - mY", output[0] - mY[0])
            # print("output", output, "mY", mY)
            loss = criterion(output, mY)
            l2_reg = 0
    
            if HP["new_acts"] == "FSig_reg":

                for j in range(len(model.lstm_layers)):
                    l2_reg = l2_reg + torch.mean((model.lstm_layers[j].p_act_layer_1.a.data - 
                        torch.mean(model.lstm_layers[j].p_act_layer_1.a.data))**2 + 
                       (model.lstm_layers[j].p_act_layer_1.b.data - 
                                             torch.mean(model.lstm_layers[j].p_act_layer_1.b.data))**2)
                
                    l2_reg = l2_reg + torch.mean((model.lstm_layers[j].p_act_layer_2.a.data - 
                        torch.mean(model.lstm_layers[j].p_act_layer_2.a.data))**2 + 
                       (model.lstm_layers[j].p_act_layer_2.b.data - 
                                             torch.mean(model.lstm_layers[j].p_act_layer_2.b.data))**2)
                
                    l2_reg = l2_reg + torch.mean((model.lstm_layers[j].p_act_layer_3.a.data - 
                        torch.mean(model.lstm_layers[j].p_act_layer_3.a.data))**2 + 
                       (model.lstm_layers[j].p_act_layer_3.b.data - 
                                             torch.mean(model.lstm_layers[j].p_act_layer_3.b.data))**2)
                
                    loss = loss + l2_reg * reg_lambda
      
            state_0 = optimizer.__getstate__()
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state_1 = optimizer.__getstate__()
            if (i+1) % 10 == 0:
                print("epoch:", epoch, "i:", i, "loss:", loss.item())
            
            i = i + 1
            
        for minibatch in minibatches_val:

            (mXv, mYv) = minibatch
            
            mXv = torch.tensor(mXv).type(torch.FloatTensor)
            if reg_class == "reg":
                mYv = torch.tensor(mYv).type(torch.FloatTensor)
            if reg_class == "class":
                mYv = torch.tensor(mYv[:,0]).type(torch.LongTensor)
                
            mXv = mXv.to(device)
            mYv = mYv.to(device)

            # Forward pass
            if HP["new_acts"] == True or HP["new_acts"] == "FSig_reg":
                output, _, __ = model(mXv)
            else:
                output = model(mXv)
                
            try:
                output = output.type(torch.FloatTensor).to(device)
            except:
                output = output[0].type(torch.FloatTensor).to(device)

            loss = criterion(output, mYv)            
            valid_losses.append(loss.item())

        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        epoch_len = len(str(num_epochs))

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
        t = time.clock()-t0
        time_list.append(t)
        
    eva_loss = avg_valid_losses[-1]
            
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        
        loss_t = 0
        
        if reg_class == "reg":
            
            for minibatch in minibatches_test:
            
                (mXt, mYt) = minibatch
                mXt = torch.tensor(mXt).type(torch.FloatTensor)
                mYt = torch.tensor(mYt).type(torch.FloatTensor)
                
                mXt = mXt.to(device)
                mYt = mYt.to(device)
            
                if args["new_acts"] == True or args["new_acts"] == "FSig_reg":
                    output, _, __ = model(mXt)
                else:
                    output = model(mXt)

                mloss = criterion(output, mYt)
                loss_t = loss_t + mloss /len(minibatches_test) 
            
            eva = loss_t
        
        if reg_class == "class":
        
            correct = 0
            total = 0
        
            for minibatch in minibatches_test:
            
                (mXt, mYt) = minibatch
                mXt = torch.tensor(mXt).type(torch.FloatTensor)
                mYt = torch.tensor(mYt[:,0]).type(torch.LongTensor)
                
                mXt = mXt.to(device)
                mYt = mYt.to(device)
                
                if args["new_acts"] == True or args["new_acts"] == "FSig_reg":
                    output, _, __ = model(mXt)
                else:
                    output = model(mXt)
                
                output = output.to(device)
                mloss = criterion(output, mYt)
                loss_t = loss_t + mloss /len(minibatches_test) 
                predicted = torch.round(output.data) 
            
                _, predicted = torch.max(output.data, 1)
                predicted = predicted.to(device)
                total += mYt.size(0)
                correct += (predicted == mYt[:].type(torch.LongTensor).to(device)).sum().item()

            print('Accuracy of the network on test data: {} %'.format(100 * correct / total))
            eva = 100 * correct / total
        
    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')

    if args["new_acts"] == True or args["new_acts"] == "FSig_reg":
        return eva, model, avg_train_losses, avg_valid_losses, time_list, act_para_list_list, act_para_std_list_list
    if args["new_acts"] == False:
        return eva, model, avg_train_losses, avg_valid_losses, time_list
    
def main_train_autoencoder(data, args):
    
    print("args", args)
    
    train_loader, val_loader, test_loader = data
    model_type = args["model_type"]
    batch_size = args["batch_size"]
    num_epochs = args["num_epochs"]
    learning_rate = args["learning_rate"]
    noise_level = args["noise_level"]
    patience = args["patience"]
    reg_lambda = args["reg_lambda"]
    reg_dev = args["reg_dev"]
    
    if model_type == "autoencoder_1":
        model = autoencoder_1(args).to(device)
    if model_type == "autoencoder_2":
        model = autoencoder_2(args).to(device)
    if model_type == "autoencoder_3":
        model = autoencoder_3(args).to(device)
    if model_type == "conv_autoencoder":
        model = conv_autoencoder(args).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = [] 
    time_list = []
    
    for epoch in range(num_epochs):
        
        for data in train_loader:
            img, _ = data
            
            if noise_level != None:
                noise = torch.rand(len(img), 1, 28, 28)
                img_n = torch.mul(img+0.25, noise_level * noise)
                
            if model_type == "autoencoder_1" or model_type == "autoencoder_2" or model_type == "autoencoder_3":
                img = img.view(img.size(0), -1)
                if noise_level != None:
                    img_n = img_n.view(img_n.size(0), -1)
                
            img = Variable(img).to(device)
            
            """
            image = img[0].cpu()
            origin = image.data.numpy()
            plt.imshow(origin[0],cmap='gray')
            plt.show()
            """
            
            if noise_level == None:
                output = model(img)
            else:
                img_n = Variable(img_n).to(device) 
                output = model(img, img_n)
            
            loss = criterion(output, img)
            # print("loss_0:", loss)
            
            l2_reg = 0
            dev = 0
            
            if args["flexible_act"] == True or args["flexible_act"] == "FReLu_reg" or args["flexible_act"] == "FTanh_reg":

                l2_reg += torch.sum((model.p_act_layer_1.a.data - torch.mean(model.p_act_layer_1.a.data))**2 
                       + (model.p_act_layer_1.b.data - torch.mean(model.p_act_layer_1.b.data))**2)
                
                l2_reg += torch.sum((model.p_act_layer_2.a.data - torch.mean(model.p_act_layer_2.a.data))**2 
                       + (model.p_act_layer_2.b.data - torch.mean(model.p_act_layer_2.b.data))**2)
                
                loss = loss + l2_reg * reg_lambda + reg_dev * dev
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
        for data in val_loader: 
            
            img, _ = data
            if model_type == "autoencoder_1" or model_type == "autoencoder_2" or model_type == "autoencoder_3":
                img = img.view(img.size(0), -1)
                
            img = Variable(img).to(device)
            output = model(img)            
            loss = criterion(output, img)
            valid_losses.append(loss.item())
            
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        train_losses = []
        valid_losses = []
            
        print('epoch [{}/{}],  train loss:{:.4f}, valid loss:{:.4f}'
          .format(epoch + 1, num_epochs, train_loss, valid_loss))
        
        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    with torch.no_grad():
        
        L = len(test_loader)
        loss_ave = 0
        
        for data in test_loader:
            
            img, _ = data
            if model_type == "autoencoder_1" or model_type == "autoencoder_2" or model_type == "autoencoder_3":
                img = img.view(img.size(0), -1)
                
            img = Variable(img).to(device)
            output = model(img)            
            loss = criterion(output, img)
            loss_ave = loss_ave + loss/L
    
    eva = loss_ave
    
    return eva, model, avg_train_losses, avg_valid_losses, time_list
    
def model_train_cnn(data, args):

    model_type = args["model_type"]
    learning_rate = args["learning_rate"]
    num_epochs = args["num_epochs"]
    dataset = args["dataset"]
    reg_lambda= args["reg_lambda"]
    patience = args["patience"]
    reg_dev = args["reg_dev"]
    print("dataset:", dataset)
    train_loader, val_loader, test_loader = data

    if model_type == "FFNN":
        model = FFNN(args).to(device)
    if model_type == "CNN":
        model = CNN(args).to(device)
    if model_type == "LeNet":
        print("LeNet!")
        model = LeNet(args).to(device)
    if model_type == "LSTM":
        model = LSTM(args).to(device)
        sequence_length = args['sequence_length']
        input_size = args["layer_size"][0]
    if model_type == "ResNet":
        model = ResNet(ResidualBlock, [2, 2, 2]).to(device)
        
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
    # optimizer = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
    
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = [] 
    time_list = []
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    t0 = time.clock()
    time_list.append(t0)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        # mini-batch training with data from train_loader
        for i, (images, labels) in enumerate(train_loader):  
            
            # Move tensors to the configured device
            if model_type == "FFNN":
                if dataset == "MNIST":
                    images = images.reshape(-1, 28*28).to(device)
                if dataset == "CIFAR10":
                    images = images.reshape(-1, 3*32*32).to(device)
            if model_type == "LSTM":
                images = images.reshape(-1, sequence_length, input_size).to(device)
                # print("images.shape", images.shape)
                # print("labels.shape", labels.shape)
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            output = model(images)
            if model_type == "FFNN":
                output = output[-1]
                # print("output", output)
            loss = criterion(output, labels)
            
            l2_reg = 0
            dev = 0
            
            if args["flexible_act"] == "PReLu_reg":
                
                l2_reg += torch.mean((model.p_act_layer_2.a.data - torch.mean(model.p_act_layer_2.a.data))**2) 
                dev = torch.mean((model.p_act_layer_2.a.data - 0) ** 2)
                loss = loss + l2_reg * reg_lambda + reg_dev * dev                    
            
            """
            if args["flexible_act"] == True:
                # print("flexible_act!")
                
                
                #l2_reg += torch.sum((model.p_act_layer_1.a.data - torch.mean(model.p_act_layer_1.a.data))**2 
                #       + (model.p_act_layer_1.b.data - torch.mean(model.p_act_layer_1.b.data))**2)
                #
                
                # print("torch.mean(model.p_act_layer_3.a.data, torch.mean(model.p_act_layer_3.b.data)", torch.mean(model.p_act_layer_3.a.data), torch.mean(model.p_act_layer_3.b.data))    
                l2_reg += torch.sum((model.p_act_layer_2.a.data - torch.mean(model.p_act_layer_2.a.data))**2 
                       + (model.p_act_layer_2.b.data - torch.mean(model.p_act_layer_2.b.data))**2)
                
                # print("torch.mean(model.p_act_layer_4.a.data, torch.mean(model.p_act_layer_4.b.data)", torch.mean(model.p_act_layer_4.a.data), torch.mean(model.p_act_layer_4.b.data))    
                
                # dev = (torch.mean(model.p_act_layer_1.a.data) - 1) ** 2 + (torch.mean(model.p_act_layer_2.a.data)-1) ** 2
                dev = torch.mean((model.p_act_layer_2.a.data - 1)**2)
                
                loss = loss + l2_reg * reg_lambda + reg_dev * dev
                """
        
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if (i+1) % 10 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            
            train_losses.append(loss.item())
        
        for i, (images, labels) in enumerate(val_loader):  
            
            # Move tensors to the configured device
            if model_type == "FFNN":
                if dataset == "MNIST":
                    images = images.reshape(-1, 28*28).to(device)
                if dataset == "CIFAR10":
                    # print("images.shape", images.shape)
                    images = images.reshape(-1, 3*32*32).to(device)
                    # print("images.shape", images.shape)
            if model_type == "LSTM":
                images = images.reshape(-1, sequence_length, input_size).to(device)
                # print("images.shape", images.shape)
                # print("labels.shape", labels.shape)
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            # print("images.shape", images.shape)
            output = model(images)
            
            if model_type == "FFNN":
                output = output[-1]
            
            # calculate the loss
            loss = criterion(output, labels)
            # record validation loss
            
            valid_losses.append(loss.item())

        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        epoch_len = len(str(num_epochs))
        
        t = time.clock()-t0
        time_list.append(t)
        
    # Test the model
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            if model_type == "FFNN":
                if dataset == "MNIST":
                    images = images.reshape(-1, 28*28).to(device)
                if dataset == "CIFAR10":
                    images = images.reshape(-1, 32*32*3).to(device)
            if model_type == "LSTM":
                images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            images = images.to(device)
            output = model(images)
            if model_type == "FFNN":
                output = output[-1]
                
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
        eva = 100 * correct / total
    
    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')

    return eva, model, train_losses, valid_losses, time_list
