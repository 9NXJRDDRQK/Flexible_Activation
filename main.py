def model_train_rnn(X_train, Y_train, X_val, Y_val, X_test, Y_test, HP):

    learning_rate = HP["learning_rate"]
    num_epochs = HP["epochs"]
    batch_size = HP["batch_size"]
    reg_lambda= HP["reg_lam"]
    reg_class = HP["reg_class"]
    patience = HP["patience"]
    reg_model = 0 # HP["reg_model"]
    
    HP1 = copy.deepcopy(HP)
    # print("HP1", HP1)
    model = LSTM(HP1).to(device)
    seed = random.randint(1,1000)
    # seed = HP["seed"]
    update = False

    # Loss and optimizer
    if reg_class == "class":
        criterion = nn.CrossEntropyLoss()
    if reg_class == "reg":
        criterion = nn.MSELoss()
    
    # optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay= reg_model)  
    
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
            # print("mX.shape, mY.shape", mX.shape, mY.shape)
            
            mX = torch.tensor(mX).type(torch.FloatTensor)
            if reg_class == "reg":
                mY = torch.tensor(mY).type(torch.FloatTensor)
            if reg_class == "class":
                mY = torch.tensor(mY[:,0]).type(torch.LongTensor)
                
            mX = mX.to(device)
            mY = mY.to(device)
            # print("mX, mY", mX.shape, mY.shape)
            
            # Forward pass
            if HP["new_acts"] == True or HP["new_acts"] == "FSig_reg":
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
    
            #"""
            if HP["new_acts"] == "FSig_reg":
                # print("FSig_reg!!!")
                # print("loss_1:", loss)
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
                # print("loss_2:", loss)
            #"""
      
            state_0 = optimizer.__getstate__()
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state_1 = optimizer.__getstate__()
            # print("state", state)
            # optimizer.step()
            # model_1 = copy.deepcopy(model)
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
            # print("mX, mY", mX.shape, mY)
            # Forward pass
            if HP["new_acts"] == True or HP["new_acts"] == "FSig_reg":
                output, _, __ = model(mXv)
            else:
                output = model(mXv)
                
            try:
                output = output.type(torch.FloatTensor).to(device)
            except:
                output = output[0].type(torch.FloatTensor).to(device)
            # output_final = output_list[-1]
            # print("output_final, mY", output, mY)
            # print("output", output[0], "mY", mY[0], "output - mY", output[0] - mY[0])
            # print("output", output, "mY", mY)
            loss = criterion(output, mYv)
            
            valid_losses.append(loss.item())

        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        epoch_len = len(str(num_epochs))
        
        # print(print_msg)
        
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
            
                if HP["new_acts"] == True or HP["new_acts"] == "FSig_reg":
                    output, _, __ = model(mXt)
                else:
                    output = model(mXt)
                
                # output = model(mXt)
                # output = output.to(device)
                
                # print("output.shape, mYt.shape", output.shape, mYt.shape)
                mloss = criterion(output, mYt)
                loss_t = loss_t + mloss /len(minibatches_test) 
            
            eva = loss_t
            # print("eva", eva)
        
        if reg_class == "class":
        
            correct = 0
            total = 0
        
            for minibatch in minibatches_test:
            
                (mXt, mYt) = minibatch
                mXt = torch.tensor(mXt).type(torch.FloatTensor)
                mYt = torch.tensor(mYt[:,0]).type(torch.LongTensor)
                
                mXt = mXt.to(device)
                mYt = mYt.to(device)
                
                # output = model(mXt)
                if HP["new_acts"] == True or HP["new_acts"] == "FSig_reg":
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

    if HP["new_acts"] == True or HP["new_acts"] == "FSig_reg":
        return eva, model, avg_train_losses, avg_valid_losses, time_list, act_para_list_list, act_para_std_list_list
    if HP["new_acts"] == False:
        return eva, model, avg_train_losses, avg_valid_losses, time_list
