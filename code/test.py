# import numpy as np
# from sklearn import preprocessing
# enc=preprocessing.OneHotEncoder()
# enc.fit([['a',0,3],['1',1,0],['0',2,1],['1',0,2]])
# list=enc.transform([[0,1,1],[1,1,0]]).toarray()
# list=np.array(list,dtype=int)
# print list
import numpy as np
a=[[1,2,3,4,5,6],[2,3,4,5,5,8],[6,4,3,2,4,6]]
b=[1,2,3,4,5,6]
a=np.array(a)
b=np.array(b)
print a[2,:]*b