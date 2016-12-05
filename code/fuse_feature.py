import os
import numpy as np

root_file='../result/'
subfile=['rf_test_result_rawdata.csv','test_result_rawdata.csv','rf_test_result_logdata.csv',\
         'test_result_logdata.csv','rf_test_result_sqrtdata.csv','test_result_sqrtdata.csv',\
         'rf_test_result_powdata.csv','test_result_powdata.csv']
feature_list=[]
for file in subfile:
    file_path=os.path.join(root_file,file)
    print file
    with open(file_path,'r') as fp:
        lines=fp.readlines()
        sub_feature=[]
        for line in lines:
            if line.split(',')[0]=='Id':
                continue
            else:
                sub_feature.append(line.strip().split(',')[1])
        feature_list.append(sub_feature)
feature_list=np.array(feature_list,dtype=np.float16).T
np.save('../tmp_output/feature_list.npy',feature_list)
