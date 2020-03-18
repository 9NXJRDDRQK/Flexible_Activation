import numpy as np
import pandas as pd
import math
import random
import copy
import time
from time import gmtime, strftime
from modules import *
from activations import *

def HP_search_rand(train_X, train_Y, val_X, val_Y, test_X, test_Y, HP, HPcv, HHP, d_HP = None, parameters = None):

    val_cost_min = 1000
    HP0 = copy.deepcopy(HP)
    best_HP = copy.deepcopy(HP)
    best_para = copy.deepcopy(parameters)
    n_sample = HHP["n_sample"]
    para_iter = HHP["para_iter"]
    iteration = HHP["iteration"]
    random = HHP["random"]
    
    t0 = time.clock()
    cost_t = []
    cost_min_list = []
    HP_random = {}
    
    hp_opt_list = []
    val_cost_list = []
    
    for hp in HPcv.keys():

        if iteration==False or d_HP==None:
            r = HPcv[str(hp)]
            # HP_random[str(hp)] = np.random.uniform(low = r[0], high = r[1], size = n_sample)
            if random == False:
                HP_random[str(hp)] = r # random.sample(population = set(r), k = n_sample)
            else:
                HP_random[str(hp)] = random.sample(population = set(r), k = n_sample)
            # HP_random[str(hp)] = np.random.choice(a = r, size = n_sample, replace = False, p=None)
            
        else:
            pp = d_HP[str(hp)]
            mu = pp[0]
            sigma = pp[1]
            r = HPcv[str(hp)]
            HP_random[str(hp)] = random.uniform(low = max(0, mu - sigma), high = (mu + sigma), size = n_sample) #*r

    print("HP_random[str(hp)]", HP_random[str(hp)])
    for i in range(n_sample):

        HP = copy.deepcopy(HP0)

        if iteration == True:
            HP = copy.deepcopy(best_HP)

        para = None

        if para_iter==True:
            para = copy.deepcopy(best_para)

        print_dict = {}

        for hp in HPcv.keys():

            hp_rand = HP_random[str(hp)]
            HP[str(hp)] = hp_rand[i]

            print_dict[str(hp)] = HP[str(hp)]
        
        print(print_dict)
        
        if HP["new_acts"] == False:
            eva, model, avg_train_losses, avg_valid_losses, time_list = model_train_rnn(train_X, train_Y, val_X, val_Y, test_X, test_Y, HP = HP)
        else:
            eva, model, avg_train_losses, avg_valid_losses, time_list, act_para_list_list, act_para_std_list_list = model_train_rnn(train_X, train_Y, val_X, val_Y, test_X, test_Y, HP = HP)
            
        val_cost = eva.item()
        
        hp_opt_list.append(print_dict)
        val_cost_list.append(val_cost)

        if val_cost < val_cost_min:

            val_cost_min = val_cost
            best_para = copy.deepcopy(parameters)
            best_HP1 = copy.deepcopy(best_HP)
            best_HP = copy.deepcopy(HP)
            print("val_cost_min",val_cost_min)
            t = time.clock() - t0
            cost_t.append(t)
            cost_min_list.append(val_cost_min)
        else:
            continue

    d_HP={}

    for hp in HPcv.keys():

        bhp = best_HP[str(hp)]
        dab = abs(best_HP[str(hp)] - best_HP1[str(hp)])
        d_HP[str(hp)] = [bhp, dab]

    return val_cost_min, best_HP, best_para, HPcv, d_HP, hp_opt_list, val_cost_list

def to_onehot(target_mat):
    
    target_mat_1 = np.zeros((len(target_mat), 2))
    target_mat_1[np.arange(len(target_mat)), target_mat[:,0]] = 1
    
    return target_mat_1

def get_mul_var_data(url_list):
    
    series_list = []
    
    for url in url_list:
        
        data = pd.read_csv(url,index_col='Date',parse_dates=True, dayfirst=False)
        begin = "2009-01-02"
        end = "2018-06-1"
        data_index = data.index
        data.index = data.index.to_period(freq="D")
        # data['Open_1'] = pd.to_numeric(data["Open"])
        for i in range(data.shape[0]):
            try:
                data['Price'][i] = pd.to_numeric(data['Price'][i].replace(',',''))
            except:
                print("url:", url)
                # print("data['Open'][i]", data['Open'][i])
    
        data['Price_1'] = data['Price'].shift(-1)
        if url!= "../Datasets/00_Benchmarks/G20_Indices/GSPC (US).csv":
            data['return'] = (data['Price']-data['Price_1'])/data['Price_1']
        else:
            data['return'] = (data['Price_1'] - data['Price'])/data['Price']
        # data['return'] = (data['Open_1'] - data['Open'])/data['Open']
        # data = float(data["Change %"].strip('%')) / 100.0
        data = data['return']
        if url!= "../Datasets/00_Benchmarks/G20_Indices/GSPC (US).csv":
            data = data.iloc[::-1]
        data = data[begin:end]
        series_list.append(data)
        df = pd.concat(series_list, axis=1)
        
    return df

def data_process_TS(data, examples, y_examples):

    nb_samples = len(data) - examples - y_examples

    # input - 2 features
    input_list = [np.expand_dims(np.atleast_2d(data[i:examples+i,:]), axis=0) for i in range(nb_samples)]
    # print(input_list)
    input_mat = np.concatenate(input_list, axis=0)

    # target - the first column in merged dataframe
    target_list = [np.atleast_2d(data[i+examples:examples+i+y_examples, :]) for i in range(nb_samples)]
    target_mat = np.concatenate(target_list, axis=0)

    return input_mat, target_mat

def data_process_panel_to_TS(data, predictors, response, examples, y_example, standard = True):
    
    nb_samples = len(data) - examples - y_examples
    data_input = data[predictors]
    data_target = data[response]
    data_input = np.array(data_input)
    data_target = np.array(data_target)
    if standard == True:
        data_input = standardize(data_input, stype = "Z", ax = 0)
    input_list = [np.expand_dims(np.atleast_2d(data_input[i:examples+i,:]), axis=0) for i in range(nb_samples)]
    input_mat = np.concatenate(input_list, axis=0)
    target_list = [np.atleast_2d(data_target[i+examples:examples+i+y_examples]) for i in range(nb_samples)]
    target_mat = np.concatenate(target_list, axis=0)
    
    return input_mat, target_mat

def split(data, frac):

    n_1 = int(len(data)*frac)
    data_1 = data[:n_1,:]
    data_2 = data[n_1:,:]

    return data_1, data_2

def random_mini_batches(X, Y, mini_batch_size, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[0]                  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)

    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :].reshape((m, Y.shape[1]))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k*mini_batch_size : (k+1)*mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k*mini_batch_size : (k+1)*mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches*mini_batch_size:, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches*mini_batch_size:, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches
