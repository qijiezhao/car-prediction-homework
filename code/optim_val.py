import numpy as np
import scipy as sp
import math
from sklearn import cross_validation
from utils import load_train_data,load_test_data
import os
import xgboost as xgb
import csv
from IPython import embed
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

# def write_pred(file_path,list_pred):
#     fw=open(file_path,'w')
#     for i in list_pred:
#         fw.write(str(i)+'\n')

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
    fuse_list,id_list,label_list,len_digit=load_train_data(file='../data/train.csv',label_trans=label_trans)
    len_sample=len(fuse_list)
    LOL=cross_validation.KFold(len_sample,n_folds=2,shuffle=False)
    #x_train,x_test,y_train,y_test=train_test_split(fuse_list,label_list,test_size=0.2)

    mse_=[]
    mse_1=[]
    mse_2=[]
    for train_index,test_index in LOL:
        x_train,x_test=fuse_list[train_index],fuse_list[test_index]
        y_train,y_test=label_list[train_index],label_list[test_index]
        xg_train=xgb.DMatrix(x_train,label=y_train)
        xg_test=xgb.DMatrix(x_test,label=y_test)

        watchlist=[(xg_train,'train'),(xg_test,'test')]
        num_round=2000
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

        regr_rf=RandomForestRegressor(max_depth=10,random_state=2)
        regr_rf.fit(x_train,y_train)
        pred1=regr_rf.predict(x_test)
        #bst1=xgb.train(param,xg_train,num_boost_round=num_round,evals=watchlist)
        #pred1=bst1.predict(xg_test)
        if label_trans==1:
            pred1=np.power(math.e,pred1)
            y_test=np.power(math.e,y_test)
        elif label_trans==2:
            pred1=pred1**2
            y_test=y_test**2
        elif label_trans==3:
            pred1=pred1**0.5
            y_test=y_test**0.5

        #mse1=MSE(pred1,y_test)
        #regression problem does not need to compute error rate.
        #error_rate=sum(int(pred[i])!=y_test[i] for i in range(len(y_test)))/float(len(y_test))
        #print 'error rate is: '+str(error_rate)


        mse1=MSE(pred1,y_test)
        mse_1.append(mse1)
        #mse=MSE(pred,y_test)

    print 'predict done! '
    print np.mean(mse_1)
    #print np.mean(mse_2)
    #print 'average precision of 2 folds: '+str(np.mean(mse))



    # regr_rf=RandomForestRegressor(max_depth=8,random_state=2)
    # regr_rf.fit(x_train,y_train)
    # pred5=regr_rf.predict(x_test)
    # mse5=MSE(pred5,y_test)
    # print 'Random Forest squared error is: '+str(mse5)
    # pred_fuse=(pred1+pred2+0.5*pred5)/2.5
    # mse_fuse=MSE(pred_fuse,y_test)
    # print 'fused model squared error is: '+str(mse_fuse)



    # regr_ada=AdaBoostRegressor(DecisionTreeRegressor(max_depth=1),n_estimators=100,random_state=np.random.RandomState(1))
    # regr_ada.fit(x_train,y_train)
    # pred6=regr_ada.predict(x_test)
    # mse6=MSE(pred6,y_test)
    # print 'Adaboost squared error is: '+str(mse6)
    #
    # for i in [0.3,0.5,1,2,3,4]:
    #     pred_fuse=(pred1+pred2+0.5*pred5+i*pred6)/float(2.5+i)
    #     mse_fuse=MSE(pred_fuse,y_test)
    #     print 'fused_2 model squared error is: '+str(mse_fuse)


    # lr=LogisticRegression()
    # lr.fit(x_train,y_train)
    # pred4=lr.predict(x_test)
    # mse4=MSE(pred4,y_test)
    # print 'logistic regression squared error is: '+str(mse4)
    # pred_fuse=(pred1+pred2+pred4)/3.0
    # mse_fuse=MSE(pred_fuse,y_test)
    # print 'fused model squared error is: '+str(mse_fuse)


    # clf=svm.SVC().fit(x_train,y_train)
    # pred3=clf.predict(x_test)
    # mse3=MSE(pred3,y_test)
    # print 'rbf-SVM squared error is: '+str(mse3)


    # print 'now computing test_set : '
    # fuse_list_test,id_list_test,len_digit=load_test_data()
    # fuse_list_test=xgb.DMatrix(fuse_list_test)
    # pred_test=bst.predict(fuse_list_test)
    # csv_result='../result/test_result_rawdata.csv'
    # csvfile=file(csv_result,'wb')
    # writer=csv.writer(csvfile)
    # writer.writerow(['Id','Score'])
    # data=[]
    # for i in range(len(id_list_test)):
    #     id_tmp=id_list_test[i]
    #     score_tmp=pred_test[i]
    #     array_tmp=(id_tmp,score_tmp)
    #     data.append(array_tmp)
    # writer.writerows(data)
    # csvfile.close()
    # if os.path.exists('../result/test_result.csv'):
    #     print 'save done!'

