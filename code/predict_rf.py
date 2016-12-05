import numpy as np
import scipy as sp
import math
from sklearn import cross_validation
from utils import load_train_data,load_test_data
import os
import xgboost as xgb
import csv
from IPython import embed
from sklearn.ensemble import RandomForestRegressor

if __name__=='__main__':
    label_trans=3
    train_list,id_list,label_list,len_digit=load_train_data(file='../data/train.csv',label_trans=label_trans)
    len_sample=len(train_list)
    test_list,id_list_test,len_digit=load_test_data(file='../data/test.csv')

    regr_rf=RandomForestRegressor(max_depth=8,random_state=2)
    regr_rf.fit(train_list,label_list)
    pred_test=regr_rf.predict(test_list)

    if label_trans==1:
        pred_test=np.power(math.e,pred_test)
    elif label_trans==2:
        pred_test=pred_test**2
    elif label_trans==3:
        pred_test=pred_test**0.5

    csv_result='../result/rf_test_result_powdata.csv'
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
    if os.path.exists('../result/rf_test_result_powdata.csv'):
        print 'save done!'

