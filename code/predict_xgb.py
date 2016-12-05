import numpy as np
import scipy as sp
import math
from sklearn import cross_validation
from utils import load_train_data,load_test_data
import os
import xgboost as xgb
import csv
from IPython import embed
# from sklearn import svm
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import AdaBoostRegressor

def MSE(list1,list2):
    len_list1=len(list1)
    len_list2=len(list2)
    if not len_list1==len_list2:
        return 0
    sum=0
    for i in range(len_list1):
        sum+=(list1[i]-list2[i])**2
    result=math.sqrt(sum/len_list1)
    return result

if __name__=='__main__':
    label_trans=3
    train_list,id_list,label_list,len_digit=load_train_data(file='../data/train.csv',label_trans=label_trans)
    len_sample=len(train_list)
    test_list,id_list_test,len_digit=load_test_data(file='../data/test.csv')

    xg_train=xgb.DMatrix(train_list,label=label_list)
    xg_test=xgb.DMatrix(test_list)

    num_round=1000
    param={
            'eta':0.03,
            'max_depth':4,
            'gamma':1,
            'min_child_weight':5,
            'subsample':0.8,
            'save_period':0,
            'booster':'gbtree',
            #'silent':1,
            'nthread':4,
            'objective':'count:poisson',
            'silent':1
        }
    bst=xgb.train(param,xg_train,num_boost_round=num_round)
    print 'now computing test_set : '

    pred_test=bst.predict(xg_test)
    if label_trans==1:
        pred_test=np.power(math.e,pred_test)
    elif label_trans==2:
        pred_test=pred_test**2
    elif label_trans==3:
        pred_test=pred_test**0.5
    csv_result='../result/test_result_powdata.csv'
    csvfile=file(csv_result,'wb')
    writer=csv.writer(csvfile)
    writer.writerow(['Id','Score'])
    data=[]
    for i in range(len(id_list_test)):
        id_tmp=id_list_test[i]
        score_tmp=pred_test[i]
        array_tmp=(id_tmp,score_tmp)
        data.append(array_tmp)
    writer.writerows(data)
    csvfile.close()
    if os.path.exists('../result/test_result_powdata.csv'):
        print 'save done!'