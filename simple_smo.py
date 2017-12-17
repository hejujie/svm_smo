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
    df1 = pd.read_csv(filename)
    return np.array(df1)
    
def shuffle_sample(data, ratio):
    '''
    Input: data to be divide, and the ratio of dev data
    Output: train data and dev data
    '''
    population = data.shape[0]
    index_all = np.array(range(data.shape[0]))
    random.seed(10)
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
data = read_data()
X_test = read_data('../input/test_clean.csv')
X_test = X_test[0:10]
data, vdata  = shuffle_sample(data, ratio = 0.99)
vdata, _  = shuffle_sample(vdata, ratio = 0.99)
X_valid = vdata[:, 0:-1]
Y_valid = vdata[:, -1]
X = data[:, 0:-1]
Y = data[:, -1]
M = X.shape[0]
print(M, Y_valid.shape[0])


# set parameter
sigma = 0.5
C = 1
tol = 1e-4
alpha_tol = 1e-7
max_iter = 10
max_passes = 10

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
passes = 0
iters = 0
best = 0
while passes < max_passes and iters < max_iter:
    num_changed_alphas = 0
    for i in range(M):
        fx_i = fx_one(X, Y, i, alpha, b)
        E_i = fx_i - Y[i]
        
        if (Y[i] * E_i < -tol and alpha[i] < C) or (Y[i] * E_i > tol and alpha[i] > 0):
            j = get_random(0, M, i)
            fx_j = fx_one(X, Y, j, alpha, b)
            E_j = fx_j - Y[j]
            old_alpha_i = alpha[i]
            old_alpha_j = alpha[j]
            
            if Y[i] == Y[j]:
                L = np.maximum(0, old_alpha_i + old_alpha_j - C)
                H = np.minimum(C, old_alpha_i + old_alpha_j)
            else:
                L = np.maximum(0, old_alpha_j - old_alpha_i)
                H = np.minimum(C, C + old_alpha_j - old_alpha_i)
               
            if L == H:
                continue
            
            eta = 2 * cache[i, j] - cache[i, i] - cache[j, j]
            if eta >= 0:
                continue
            new_alpha_j = old_alpha_j - Y[j] * (E_i - E_j) / eta
            new_alpha_j = np.maximum(new_alpha_j, L)
            new_alpha_j = np.minimum(new_alpha_j, H)
            if np.abs(new_alpha_j - old_alpha_i) < alpha_tol:
                continue
            new_alpha_i = old_alpha_i + Y[i]*Y[j]*(old_alpha_j - new_alpha_j)
#            print(i, j)
#            print(new_alpha_i, new_alpha_j)
            alpha[i] = new_alpha_i
            alpha[j] = new_alpha_j

            b1 = b - E_i - Y[i] * (new_alpha_i - old_alpha_i) * cache[i, i] - \
                            Y[j] * (new_alpha_j - old_alpha_j) * cache[i, j]
            b2 = b - E_j - Y[i] * (new_alpha_i - old_alpha_i) * cache[i, j] - \
                            Y[j] * (new_alpha_j - old_alpha_j) * cache[j, j]
            
            # first consider if all satifity:
            b = (b1 + b2) / 2
            if new_alpha_i > 0 and new_alpha_i < C:     
                b = b1
            elif new_alpha_j > 0 and new_alpha_j < C:
                b = b2
            num_changed_alphas += 1
    iters += 1
    predict = (predict_label(X, Y, X, alpha, b))
    accuracy = cal_accuracy(predict, Y)
    print("train", accuracy)
    predict = (predict_label(X, Y, X_valid, alpha, b))
    accuracy = cal_accuracy(predict, Y_valid)
    print("valid", accuracy)
    
    predict = (predict_label(X, Y, X_test, alpha, b))
    predict[predict == -1] = 0
    output = pd.DataFrame(predict.astype('int32'))
    output.to_csv('../output/svm_' + str(iters) + '.csv', index = False, header = None)
    
    if num_changed_alphas == 0:
        passes += 1
    else:
        passes = 0


                 
print("cost time : ", time.time() - begin_time)        



#plt.plot(data[:, 0], data[:, 1], 'ro')




    

