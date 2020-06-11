# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:08:31 2020

@author: User
"""

import numpy as np
import pandas as pd
import random
from scipy.io import loadmat
words_en = loadmat('english_words.mat')
words_sp = loadmat('spanish_words.mat')
words_en = words_en["english_words"]
words_sp = words_sp["spanish_words"]
n = 500
n_train = 100
n_test = n - n_train

########Different Possible Kernel Methods##########################
def k0(s1,s2):
    total = 0
    for i in range(len(s1)):
        if s1[i]==s2[i]:
            total+=1
        else:
            continue
    return total

def k1(s1,s2):
    return set(s1).intersection(set(s2)).__len__()
    
def k2(s1,s2):
    s1_lst = []
    s2_lst = []
    for i in range(7):
        s1_lst += [s1[i:i+3]]
        s2_lst += [s2[i:i+3]]
    total = 0 
    for i in s1_lst:
        for j in range(len(s2_lst)):
            if i == s2_lst[j]:
                s2_lst[j] = ""
                total+=1
                break
    
    return total


#############Implement Kernelized K-Nearest Neighbour Using the Kernels Above#######################################
def kernelized_knn(words_en,words_sp,k, ker, true_is_sp, n_train=100, n=500):# Change to k0, k1, k2 as required#; % Set to 1 to use test data with true label 'spanish'
    num_errors = 0
    if true_is_sp==1:
        words1 = words_sp
        words2 = words_en
    else:
        words1 = words_en   
        words2 = words_sp

    def d(x,x0,k): #dist function
        return np.sqrt(k(x, x) + k(x0, x0) - 2*k(x, x0))
    wrong_lst = []
    for t in range(n_train,n):   
        s_next = words1[t] #; % This is the 'x' value of the next test data point (a string of length 9)
        lst1 = []
        lst2 = []
        for i in range(n_train):
            lst1 += [d(words1[i],s_next, ker)]
            lst2 += [d(words2[i],s_next, ker)]
        lst1.sort()
        lst2.sort()
        index = int((k+1)/2)
        if lst1[index]<lst2[index]:
            continue
        elif lst1[index]==lst2[index]:
            num_errors += 0.5
        else:
            wrong_lst += [s_next]    
            num_errors += 1
    
    return (num_errors / n_test), wrong_lst[random.randint(0,len(wrong_lst)-1)]
kernelized_knn(words_en,words_sp,5, ker = k2, true_is_sp=1, n_train=100, n=500)

for i in [k0,k1,k2]:
    for j in range(0,2):
        print("kernel function:",i,"sp" if j else "en",kernelized_knn(words_en,words_sp,5, ker = i, true_is_sp=j, n_train=100, n=500))

############randomly pick out one wrong entry and get 5 of its nearest neighbour######################
def get_knn(words_en,words_sp,word,k,ker):
    word_lst = []
    kernel_score_lang = []
    kernel_score = []
    for i in range(100,500):
        s1 = words_sp[i]
        word_lst += [s1]
        kernel_score_lang += ["(" + s1 + " , " + "(SP,{})".format(ker(word,s1)) + ")"]
        kernel_score += [ker(word,s1)]
        e1 = words_en[i]
        word_lst += [e1]
        kernel_score_lang += ["(" + e1 + " , " + "(EN,{})".format(ker(word,e1)) + ")"]
        kernel_score += [ker(word,e1)]
    df = pd.DataFrame({"word":word_lst,"ker_lang":kernel_score_lang, "score":kernel_score})
    df.sort_values("score",inplace=True, ascending = False)
    return df.ker_lang.iloc[1:6]
        
    
    
# words_en
# words_sp
for i in [k0,k1,k2]:
    for j in range(0,2):
        error, word = kernelized_knn(words_en,words_sp,5, ker = i, true_is_sp=j, n_train=100, n=500)
        if j==1:
            print(word,"(SP)")
            print(get_knn(words_en,words_sp,word,5,i))
        else:
            print(word,"(EN)")
            print(get_knn(words_en,words_sp,word,5,i))
            
            
            
