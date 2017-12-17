# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 18:15:45 2017

@author: jie
"""

import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_data(filename = '../input/train_clean.csv'):
    df1 = pd.read_csv(filename, header = None)
    return np.array(df1)
    
def shuffle_sample(data, ratio):
    '''
    Input: data to be divide, and the ratio of dev data
    Output: train data and dev data
    '''
    population = data.shape[0]
    index_all = np.array(range(data.shape[0]))
#    random.seed(10)
    index_valid = random.sample(range(population), int(ratio * population))
    index_train = np.delete(index_all, index_valid, axis = 0)
    return data[index_train], data[index_valid] 
    

def fx_one(X, Y, i, alpha, b):
    fx = 0
    for j in range(alpha.shape[0]):
        fx += alpha[j] * Y[j] * cache[i, j]
    return fx + b
    

def predict_one(X, Y, x_i, alpha, b):
    fx = 0
    for j in range(alpha.shape[0]):
        fx += alpha[j] * Y[j] * kernel(x_i, X[j])
    return fx + b

def predict_label(X, Y, X_pred, alpha, b):
    predict = np.zeros((X_pred.shape[0], 1))
    for i in range(X_pred.shape[0]):
        predict[i] = predict_one(X, Y, X_pred[i], alpha, b)
        
    predict = np.sign(predict)
    predict[predict == 0] = 1
    return predict

def cal_accuracy(predict, Y_pred):
    accuracy = np.mean(predict.T == Y_pred)
    return accuracy

    
def get_random(begin, end, exclusive):
    value = exclusive
    while value == exclusive:
        value = random.randint(begin, end-1)
    return value
 
def kernel(x_i, x_j, kernel = "RBF"):
    if kernel == "linear":
        return np.dot(x_i, x_j.T)
    elif kernel == "RBF":   
        s = np.sum(np.square(x_i - x_j))
        return np.exp(-s / (2 * sigma * sigma))
 
begin_time = time.time()
# load data
data = read_data('../input/train_simple.csv')
X_test = read_data('../input/test.csv')
print(X_test.shape)
X_test = X_test[0:10]

data, vdata  = shuffle_sample(data, ratio = 0.8)

#data, vdata  = shuffle_sample(data, ratio = 0.99)
vdata, _  = shuffle_sample(vdata, ratio = 0.99)
X_valid = vdata[:, 0:-1]
Y_valid = vdata[:, -1]
X = data[:, 0:-1]
Y = data[:, -1]
M = X.shape[0]
print(M)


# set parameter
sigma = 0.5
C = 1
tol = 1e-4
alpha_tol = 1e-7
max_iter = 2000
early_stop = 5


# linear kernel cache
# cache = np.dot(X, X.T)

#RBF kernel cache
cache = np.zeros((X.shape[0], X.shape[0]))
for i in range(X.shape[0]):
    for j in range(X.shape[0]):
        cache[i, j] = kernel(X[i], X[j])
        
print("cache out")


# SMO algorithm
## Initial
alpha = np.zeros((M, 1))
b = np.zeros((1, 1))
no_inprove = 0
iters = 0
best = 0
E = np.zeros((M, 1))
for i in range(M):
    fx_i = fx_one(X, Y, i, alpha, b)
    E[i] = fx_i - Y[i]
print("calculate E finished\n")

i_pass = {}
j_pass = {}
while iters < max_iter:

    update_flag = 0
    max_violated = 0
    I, J = -1, -1
    # pick the first element
    for i in range(M):
        fx_i = fx_one(X, Y, i, alpha, b)
        E[i] = fx_i - Y[i]
        
        G_i = Y[i] * E[i]
        if alpha[i] < C and G_i < -tol:
            violated = np.abs(-tol - G_i)
            if violated > max_violated and i not in i_pass:
                max_violated = violated
                I = i
                
        elif alpha[i] > 0 and G_i > tol:
            violated = np.abs(tol - G_i)
            if violated > max_violated and i not in i_pass:
                max_violated = violated
                I = i
    # pick the second element
    if I == -1:
        print("Not element break the KKT theorm, early stopped iteration\n")
        break
    
    i = I
    while update_flag == 0:
        delta = np.abs(E - E[i])
        max_delta = np.min(delta, axis = 0) - 1
        for j in range(delta.shape[0]):
            if delta[j] > max_delta and j != I and j not in j_pass:
                max_delta = delta[j]
                J = j
                
        if J == -1:
            j_pass.clear()
            i_pass[I] = 1
            print("All of j is work, Need to choose the first element\n")
            break
    
        j = J       
        fx_j = fx_one(X, Y, j, alpha, b)
        E[j] = fx_j - Y[j]
        old_alpha_i = alpha[i]
        old_alpha_j = alpha[j]
        
        if Y[i] == Y[j]:
            L = np.maximum(0, old_alpha_i + old_alpha_j - C)
            H = np.minimum(C, old_alpha_i + old_alpha_j)
        else:
            L = np.maximum(0, old_alpha_j - old_alpha_i)
            H = np.minimum(C, C + old_alpha_j - old_alpha_i)
       
        if L == H:
#            print("L == H for this j = {}".format(J))
            j_pass[J] = 1
            continue
    
        eta = 2 * cache[i, j] - cache[i, i] - cache[j, j]
        if eta >= 0:
            j_pass[J] = 1
            continue
        
        new_alpha_j = old_alpha_j - Y[j] * (E[i] - E[j]) / eta
        new_alpha_j = np.maximum(new_alpha_j, L)
        new_alpha_j = np.minimum(new_alpha_j, H)
        if np.abs(new_alpha_j - old_alpha_i) < alpha_tol:
            print("J is not move enough")
            j_pass[J] = 1
            continue
        
        new_alpha_i = old_alpha_i + Y[i]*Y[j]*(old_alpha_j - new_alpha_j)
        
        alpha[i] = new_alpha_i
        alpha[j] = new_alpha_j
    
        b1 = b - E[i] - Y[i] * (new_alpha_i - old_alpha_i) * cache[i, i] - \
                        Y[j] * (new_alpha_j - old_alpha_j) * cache[i, j]
        b2 = b - E[j] - Y[i] * (new_alpha_i - old_alpha_i) * cache[i, j] - \
                        Y[j] * (new_alpha_j - old_alpha_j) * cache[j, j]
        
        # first consider if all satifity:
        b = (b1 + b2) / 2
        if new_alpha_i > 0 and new_alpha_i < C:     
            b = b1
        elif new_alpha_j > 0 and new_alpha_j < C:
            b = b2
        
        if iters % 100 == 0:
            print("iteration is: {}".format(iters))
        iters += 1
        update_flag = 1
        i_pass.clear()
        j_pass.clear()
    
    if iters % 10 == 0:
        predict = (predict_label(X, Y, X, alpha, b))
        accuracy = cal_accuracy(predict, Y)
        print("train", accuracy)
        last = accuracy
        if np.abs(last - accuracy) < 1e-2:
            no_inprove += 1
        if no_inprove >= early_stop:
            print("early stop because not improve")
            break

#        predict = (predict_label(X, Y, X_valid, alpha, b))
#        accuracy = cal_accuracy(predict, Y_valid)
#        print("valid", accuracy)
        
        
#        predict = (predict_label(X, Y, X_test, alpha, b))
#        predict[predict == -1] = 0
#        output = pd.DataFrame(predict.astype('int32'))
#        output.to_csv('../output/svm_' + str(iters) + '.csv', index = False, header = None)



                 
print("cost time : ", time.time() - begin_time)        



#plt.plot(data[:, 0], data[:, 1], 'ro')




    

