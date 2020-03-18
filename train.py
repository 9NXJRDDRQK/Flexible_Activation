import numpy as np
import pandas as pd
import random
import copy

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn.functional as F
from torch.autograd import *
import math
import time
import matplotlib.pyplot as plt
from time import gmtime, strftime
from torch.autograd.variable import Variable

import modules
import activations

def train_autoencoder(data, args):
    
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
    # para_group_name_list = args["para_group_name_list"]
    # reg_model = 0 # HP["reg_model"]
    
    if model_type == "autoencoder_1":
        model = autoencoder_1(args).to(device)
    if model_type == "autoencoder_2":
        model = autoencoder_2(args).to(device)
    if model_type == "autoencoder_3":
        model = autoencoder_3(args).to(device)
    if model_type == "conv_autoencoder":
        model = conv_autoencoder(args).to(device)
    if model_type == "conv_autoencoder_cifar":
        model = conv_autoencoder_cifar(args).to(device)
    
    criterion = nn.MSELoss()
    
    """
    para_namelist = list(model.state_dict().keys())
    optimizer_list = []
   
    for i in range(len(para_group_name_list)):
        optimizer = Adam_HD_2(model.parameters(), para_namelist, para_group_name_list[i], lr=learning_rate)
        optimizer_list.append(optimizer)
    """
    
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
           
            if noise_level == None:
                output = model(img)
            else:
                img_n = Variable(img_n).to(device) 
                output = model(img, img_n)
            
            loss = criterion(output, img)
            
            l2_reg = 0
            dev = 0
            l2_reg_beta = 0
            
            if args["flexible_act"] == True or args["flexible_act"] == "FReLu_reg" or args["flexible_act"] == "FTanh_reg" or args["flexible_act"] == "Relu_Elu_reg":

                l2_reg += torch.sum((model.p_act_layer_1.a - torch.mean(model.p_act_layer_1.a))**2 
                       + (model.p_act_layer_1.b - torch.mean(model.p_act_layer_1.b))**2)
                
                l2_reg += torch.sum((model.p_act_layer_2.a - torch.mean(model.p_act_layer_2.a))**2 
                       + (model.p_act_layer_2.b - torch.mean(model.p_act_layer_2.b))**2)
                
                l2_reg += torch.sum((model.p_act_layer_3.a - torch.mean(model.p_act_layer_3.a))**2 
                       + (model.p_act_layer_3.b - torch.mean(model.p_act_layer_3.b))**2)
                
                l2_reg += torch.sum((model.p_act_layer_4.a - torch.mean(model.p_act_layer_4.a))**2 
                       + (model.p_act_layer_4.b - torch.mean(model.p_act_layer_4.b))**2)
                
                l2_reg += torch.sum((model.p_act_layer_5.a - torch.mean(model.p_act_layer_5.a))**2 
                       + (model.p_act_layer_5.b - torch.mean(model.p_act_layer_5.b))**2)
                    
                l2_reg += torch.sum((model.p_act_layer_6.a - torch.mean(model.p_act_layer_6.a))**2 
                       + (model.p_act_layer_6.b - torch.mean(model.p_act_layer_6.b))**2)

                loss = loss + l2_reg * reg_lambda + reg_dev * dev
                
            if args["flexible_act"] == True or args["flexible_act"] == "FReLu_reg_1" or args["flexible_act"] == "FTanh_reg_1" or args["flexible_act"] == "Relu_Elu_reg_1":

                l2_reg += torch.sum((model.p_act_layer_1.a - 1)**2 
                       + (model.p_act_layer_1.b - 0)**2)
                
                l2_reg += torch.sum((model.p_act_layer_2.a - 1)**2 
                       + (model.p_act_layer_2.b - 0)**2)
                
                l2_reg += torch.sum((model.p_act_layer_3.a - 1)**2 
                       + (model.p_act_layer_3.b - 0)**2)
                
                l2_reg += torch.sum((model.p_act_layer_4.a - 1)**2 
                       + (model.p_act_layer_4.b - 0)**2)
                
                l2_reg += torch.sum((model.p_act_layer_5.a - 1)**2 
                       + (model.p_act_layer_5.b - 0)**2)
                
                l2_reg += torch.sum((model.p_act_layer_6.a - 1)**2 
                       + (model.p_act_layer_6.b - 0)**2)
                
                print("l2_reg * reg_lambda 2:", l2_reg * reg_lambda)
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

def model_train_rnn(X_train, Y_train, X_val, Y_val, X_test, Y_test, HP):

    learning_rate = HP["learning_rate"]
    num_epochs = HP["epochs"]
    batch_size = HP["batch_size"]
    reg_lambda= HP["reg_lam"]
    reg_class = HP["reg_class"]
    patience = HP["patience"]
    # para_group_name_list = HP["para_group_name_list"]
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

    # optimizer = Adam(model.parameters(), args)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay= reg_model)  
        
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
        
        # print("len(minibatches), len(minibatches_val)", len(minibatches), len(minibatches_val))
        
        i = 0
        for minibatch in minibatches:

            (mX, mY) = minibatch
            # print("mX.shape, mY.shape", mX.shape, mY.shape)
            
            mX = torch.tensor(mX).type(torch.FloatTensor)
            # print("mX.shape", mX.shape)
            
            if reg_class == "reg":
                mY = torch.tensor(mY).type(torch.FloatTensor)
            if reg_class == "class":
                mY = torch.tensor(mY[:,0]).type(torch.LongTensor)
                
            mX = mX.to(device)
            mY = mY.to(device)
            # print("mX, mY", mX.shape, mY.shape)
            
            # Forward pass
            if HP["new_acts"] == True or HP["new_acts"] == "FSig_reg" or HP["new_acts"] == "FSig_reg_1":
                # print("new_acts!!!")
                output, act_para_list, act_para_std_list = model(mX)
                act_para_list_list.append(act_para_list)
                act_para_std_list_list.append(act_para_std_list)
            else:
                output = model(mX)
                
            # print("output.shape, mY.shape", output.shape, mY.shape)
            loss = criterion(output, mY)
            l2_reg = 0
            # print("loss_0:", loss.item())
    
            # """
            if HP["new_acts"] == "FSig_reg":
                # print("FSig_reg!!!")
                # print("loss_1:", loss)
                for j in range(len(model.lstm_layers)):
                    l2_reg = l2_reg + torch.mean((model.lstm_layers[j].p_act_layer_1.a - 
                        torch.mean(model.lstm_layers[j].p_act_layer_1.a))**2 + 
                       (model.lstm_layers[j].p_act_layer_1.b - 
                                             torch.mean(model.lstm_layers[j].p_act_layer_1.b))**2)
                
                    l2_reg = l2_reg + torch.mean((model.lstm_layers[j].p_act_layer_2.a - 
                        torch.mean(model.lstm_layers[j].p_act_layer_2.a))**2 + 
                       (model.lstm_layers[j].p_act_layer_2.b - 
                                             torch.mean(model.lstm_layers[j].p_act_layer_2.b))**2)
                
                    l2_reg = l2_reg + torch.mean((model.lstm_layers[j].p_act_layer_3.a - 
                        torch.mean(model.lstm_layers[j].p_act_layer_3.a))**2 + 
                       (model.lstm_layers[j].p_act_layer_3.b - 
                                             torch.mean(model.lstm_layers[j].p_act_layer_3.b))**2)
                
                loss = loss + l2_reg * reg_lambda
                
            if HP["new_acts"] == "FSig_reg_1":
                # print("FSig_reg!!!")
                # print("loss_1:", loss)
                for j in range(len(model.lstm_layers)):
                    l2_reg = l2_reg + torch.mean((model.lstm_layers[j].p_act_layer_1.a - 1)**2) 
                    l2_reg = l2_reg + torch.mean((model.lstm_layers[j].p_act_layer_2.a - 1)**2)
                    l2_reg = l2_reg + torch.mean((model.lstm_layers[j].p_act_layer_3.a - 1)**2)
                
                loss = loss + l2_reg * reg_lambda
                    
            # print("loss_1:", loss.item())
            # """
                    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            if (i+1) % 10 == 0:
                print("epoch:", epoch, "i:", i, "loss:", loss.item())
            
            i = i + 1
            
        for minibatch in minibatches_val:

            (mXv, mYv) = minibatch
            
            # print("mXv.shape, mYv", mXv.shape, mYv.shape)
            
            mXv = torch.tensor(mXv).type(torch.FloatTensor)
            if reg_class == "reg":
                mYv = torch.tensor(mYv).type(torch.FloatTensor)
            if reg_class == "class":
                mYv = torch.tensor(mYv[:,0]).type(torch.LongTensor)
                
            mXv = mXv.to(device)
            mYv = mYv.to(device)
            # print("mX, mY", mX.shape, mY)
            # Forward pass
            if HP["new_acts"] == True or HP["new_acts"] == "FSig_reg" or HP["new_acts"] == "FSig_reg_1":
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
    
    # m_a_1 = 
    
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
            
                if HP["new_acts"] == True or HP["new_acts"] == "FSig_reg" or HP["new_acts"] == "FSig_reg_1":
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
                if HP["new_acts"] == True or HP["new_acts"] == "FSig_reg" or HP["new_acts"] == "FSig_reg_1":
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

    if HP["new_acts"] == True or HP["new_acts"] == "FSig_reg" or HP["new_acts"] == "FSig_reg_1":
        return eva, model, avg_train_losses, avg_valid_losses, time_list, act_para_list_list, act_para_std_list_list
    if HP["new_acts"] == False:
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
