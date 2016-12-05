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
    label_trans=0
    preds_val=[]
    preds_test=[]
    mse=[]
    val_list_label_fix=[]
    test_list_label_fix=[]
    fuse_list,id_list,label_list,len_digit=load_train_data(file='../data/train.csv')
    len_sample=len(fuse_list)
    for label_trans in [0,1,2,3]:
        if label_trans==1:
            label_list=np.log(label_list)
        elif label_trans==2:
            label_list=label_list**0.5
        elif label_trans==3:
            label_list=label_list**2

        train_list,train_label=fuse_list[0:20000,],label_list[0:20000]
        val_list,val_label=fuse_list[20000:28000,],label_list[20000:28000]
        test_list,test_label=fuse_list[28000:40000,],label_list[28000:40000]

        if label_trans==0:
            val_list_label_fix=val_label
            test_list_label_fix=test_label
            mean_val_label=np.mean(val_list_label_fix)
            mean_test_label=np.mean(test_list_label_fix)
            with open('../tmp_output/tmp.txt','a') as fw:
                fw.write('mean value of val label: '+str(mean_val_label)+'\n')
                fw.write('mean value of test label:'+str(mean_test_label)+'\n')

        #x_train,x_test,y_train,y_test=train_test_split(fuse_list,label_list,test_size=0.2)

        xg_train=xgb.DMatrix(train_list,label=train_label)
        xg_test=xgb.DMatrix(val_list,label=val_label)

        watchlist=[(xg_train,'train'),(xg_test,'test')]
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

        regr_rf=RandomForestRegressor(max_depth=8,random_state=2)
        regr_rf.fit(train_list,train_label)
        pred1_val=regr_rf.predict(val_list)
        pred1_test=regr_rf.predict(test_list)


        bst1=xgb.train(param,xg_train,num_boost_round=num_round,evals=watchlist)
        pred2_val=bst1.predict(xg_test)
        xg_test=xgb.DMatrix(test_list)
        pred2_test=bst1.predict(xg_test)


        if label_trans==1:
            label_list=np.power(math.e,label_list)
            pred1_val=np.power(math.e,pred1_val)
            pred2_val=np.power(math.e,pred2_val)
            pred1_test=np.power(math.e,pred1_test)
            pred2_test=np.power(math.e,pred2_test)
            test_label=np.power(math.e,test_label)

        elif label_trans==2:
            label_list=label_list**2
            pred1_val=pred1_val**2
            pred1_test=pred1_test**2
            pred2_val=pred2_val**2
            pred2_test=pred2_test**2
            test_label=test_label**2
        elif label_trans==3:
            label_list=label_list**0.5
            pred1_val=pred1_val**0.5
            pred1_test=pred1_test**0.5
            pred2_val=pred2_val**0.5
            pred2_test=pred2_test**0.5
            test_label=test_label**0.5


        preds_val.append(pred1_val)
        preds_test.append(pred1_test)
        preds_val.append(pred2_val)
        preds_test.append(pred2_test)

        mean_val_rf=np.mean(pred1_val)
        mean_val_xgb=np.mean(pred2_val)
        #mse1=MSE(pred1,y_test)
        #regression problem does not need to compute error rate.
        #error_rate=sum(int(pred[i])!=y_test[i] for i in range(len(y_test)))/float(len(y_test))
        #print 'error rate is: '+str(error_rate)
        mse1=MSE(pred1_val,val_list_label_fix)
        mse2=MSE(pred2_val,val_list_label_fix)
        mse.append(mse1)
        mse.append(mse2)

        str1='randomforest + '+str(label_trans)+' val rmse: '+str(mse1)+'| mean label value is :'+str(mean_val_rf)
        pred1_val_postmean=pred1_val*(mean_val_label/mean_val_rf)
        mse_post_rf=MSE(pred1_val_postmean,val_list_label_fix)
        str1_='rf postmean mse is :'+str(mse_post_rf)
        str2='xgboost + '+str(label_trans)+' val rmse: '+str(mse2)+'| mean label value is :'+str(mean_val_xgb)
        pred2_val_postmean=pred2_val*(mean_val_label/mean_val_xgb)
        mse_post_xgb=MSE(pred2_val_postmean,val_list_label_fix)
        str2_='xgb postmean mse is :'+str(mse_post_xgb)

        print str1
        print str2
        with open('../tmp_output/tmp.txt','a') as fw:
            fw.write(str1+'\n')
            fw.write(str1_+'\n')
            fw.write(str2+'\n')
            fw.write(str2_+'\n')

            #mse=MSE(pred,y_test)
    new_train=np.array(preds_val).T
    new_val=np.array(preds_test).T
    new_train_label=val_list_label_fix
    new_val_label=test_list_label_fix

    regr_rf=RandomForestRegressor(max_depth=8,random_state=2)
    regr_rf.fit(new_train,new_train_label)
    pred_new=regr_rf.predict(new_val)
    mse_new=MSE(pred_new,new_val_label)
    str3='2nd RF : '+str(mse_new)+' | mean value of 2nd RF is :'+str(np.mean(pred_new))
    print str3
    with open('../tmp_output/tmp.txt','a') as fw:
        fw.write(str3+'\n')
    print mse

    xg_train=xgb.DMatrix(new_train,label=new_train_label)
    xg_test=xgb.DMatrix(new_val,label=new_val_label)

    watchlist=[(xg_train,'train'),(xg_test,'test')]
    num_round=100
    param={
        'eta':0.3,
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

    bst1=xgb.train(param,xg_train,num_boost_round=num_round,evals=watchlist)
    pred_2nd_xgb=bst1.predict(xg_test)
    mse_2nd_xgb=MSE(pred_2nd_xgb,new_val_label)
    str4='xgboost 2nd : '+str(mse_2nd_xgb)+'| mean value of 2nd xgb is :'+str(np.mean(pred_2nd_xgb))
    print str4
    with open('../tmp_output/tmp.txt','a')as fw:
        fw.write(str4+'\n')

    len_new_train=len(new_train)
    fuse_new_train=np.zeros(len_new_train)
    weight_val=[0,1,2,4]
    weights_item=[]
    for num1 in weight_val[:-2]:
        for num2 in weight_val[1:]:
            for num3 in weight_val[:-2]:
                for num4 in weight_val:
                    for num5 in weight_val[:-2]:
                        for num6 in weight_val:
                            for num7 in weight_val[:-2]:
                                for num8 in weight_val:
                                    weight_item=[]
                                    weight_item.append(num1)
                                    weight_item.append(num2)
                                    weight_item.append(num3)
                                    weight_item.append(num4)
                                    weight_item.append(num5)
                                    weight_item.append(num6)
                                    weight_item.append(num7)
                                    weight_item.append(num8)
                                    weights_item.append(weight_item)
    temp_min=10
    best_item=np.ones(8)
    num_count=0
    for item in weights_item:
        num_count+=1
        if num_count%50==0:
            print num_count
        for i in range(len_new_train):
            fuse_new_train[i]=np.sum(new_train[i,:]*item)/np.sum(item)
        mse_2nd_weight=MSE(fuse_new_train,new_train_label)
        if mse_2nd_weight<temp_min:
            temp_min=mse_2nd_weight
            best_item=item
    print temp_min
    print best_item
    str5='best uniform distribution of these feature is: '+str(best_item)+'| result value is '+str(temp_min)
    with open('../tmp_output/tmp.txt','a') as fw:
        fw.write(str5+'\n')


    #prediction:
    test_data=np.load('../tmp_output/feature_list.npy')
    len_test_data=len(test_data)


    if temp_min<mse_2nd_xgb and temp_min<mse_new:
        print 'best fusion is linear method.....and prediting.....'
        fused_test_data=np.zeros(len_test_data)
        for i in range(len_test_data):
            fused_test_data[i]=np.sum(test_data[i,:]*best_item)/np.sum(best_item)

    if mse_2nd_xgb<temp_min and mse_2nd_xgb<mse_new:
        print 'best fusion is xgboost again.....and predicting.....'
        xg_test=xgb.DMatrix(test_data)
        fused_test_data=bst1.predict(xg_test)

    if mse_new<temp_min and mse_new<mse_2nd_xgb:
        print 'best fusion is randomforest.....and predicting.....'
        fused_test_data_=regr_rf.predict(test_data)


    csv_result='../result/last_fuse.csv'
    csvfile=file(csv_result,'wb')
    writer=csv.writer(csvfile)
    writer.writerow(['Id','Score'])
    data=[]
    for i in range(len_test_data):
        id_tmp=str(i+40000+1)
        score_tmp=fused_test_data[i]
        array_tmp=(id_tmp,score_tmp)
        data.append(array_tmp)
    writer.writerows(data)
    csvfile.close()
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

