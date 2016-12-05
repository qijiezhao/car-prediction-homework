import numpy as np
import scipy as sp
from sklearn import preprocessing
import math
import random
#import xgboost as xgb

train_file='../data/train.csv'
test_file='../data/test.csv'
def OHE(alpha_list,flag):#use both train+test_set to create one hot encoder method
    if flag==0:
        global enc
        enc=preprocessing.OneHotEncoder()
        alpha_lists=[]
        tmp_lines1=open(train_file,'r').readlines()
        for i in range(len(tmp_lines1)):
            if i==0:
                continue
            line_list=tmp_lines1[i].strip().split(',')
            line_list=line_list[2:]
            line_alpha=[ord(alpha) for alpha in line_list if alpha.isdigit()==False]
            alpha_lists.append(line_alpha)
        tmp_lines2=open(test_file,'r').readlines()
        for j in range(len(tmp_lines2)):
            if j==0:
                continue
            line_list=tmp_lines2[j].strip().split(',')
            line_list=line_list[1:]
            line_alpha=[ord(alpha) for alpha in line_list if alpha.isdigit()==False]
            alpha_lists.append(line_alpha)
        enc.fit(alpha_lists)
    trans_list=enc.transform(alpha_list).toarray()
    return trans_list

def load_train_data(file=train_file):
    fp=open(file,'r')
    tmp_lines=fp.readlines()
    tmp_lines=tmp_lines[1:]
    random.shuffle(tmp_lines)
    len_train=len(tmp_lines)
    digit_list=[]
    alpha_list=[]
    id_list=[]
    label_list=[]
    for i in range(len_train):
        line_list=tmp_lines[i].strip().split(',')
        id_list.append(int(line_list[0]))
        label_list.append(float(line_list[1]))
        line_list=line_list[2:]
        line_digit=[float(digit) for digit in line_list if digit.isdigit()==True]
        line_alpha=[ord(alpha) for alpha in line_list if alpha.isdigit()==False]
        len_gap=len(line_digit)
        digit_list.append(line_digit)
        alpha_list.append(line_alpha)
    #encoding:
    trans_list=OHE(alpha_list,0)
    #fuse digit and alpha list
    trans_list=np.array(trans_list,dtype=int)
    digit_list=np.array(digit_list,dtype=int)
    fuse_list=np.concatenate([digit_list,trans_list],axis=1)
    label_list=np.array(label_list)

    fp.close()
    return fuse_list,id_list,label_list,len_gap

def load_test_data(file=test_file):
    fp=open(file,'r')
    tmp_lines=fp.readlines()
    len_test=len(tmp_lines)
    digit_list=[]
    alpha_list=[]
    id_list=[]
    for i in range(len_test):
        if i==0:
            continue
        line_list=tmp_lines[i].strip().split(',')
        id_list.append(int(line_list[0]))
        line_list=line_list[1:]
        line_digit=[float(digit) for digit in line_list if digit.isdigit()==True]
        line_alpha=[ord(alpha) for alpha in line_list if alpha.isdigit()==False]
        len_gap=len(line_digit)
        digit_list.append(line_digit)
        alpha_list.append(line_alpha)
    #encoding:
    trans_list=OHE(alpha_list,1)
    #fuse digit and alpha list
    trans_list=np.array(trans_list,dtype=int)
    digit_list=np.array(digit_list,dtype=int)
    fuse_list=np.concatenate([digit_list,trans_list],axis=1)
    fp.close()
    return fuse_list,id_list,len_gap

# fuse_list,len_gap=load_data()
# print fuse_list,len_gap

