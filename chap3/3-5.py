import numpy as np

def train(dataset1,dataset0,dataset):
    data1=np.mat(dataset1)
    data0=np.mat(dataset0)
    data=np.mat(dataset)
    mean1=np.mean(data1,axis=1)
    mean0=np.mean(data0,axis=1)
    Sb=(mean1-mean0)*((mean1-mean0).T)
    m1,n1=np.shape(data1)
    m0,n0=np.shape(data0)
    delta1=data1-np.repeat(mean1,n1,1)
    delta0=data0-np.repeat(mean0,n0,1)
    for i in range(n1):
        temp=delta1[:,i]
        Sw1=temp*(temp.T)
    for i in range(n0):
        temp=delta0[:,i]
        Sw0=temp*(temp.T)
    Sw=Sw1+Sw0
    W=(Sw.T)*Sb
    e_value,e_vector=np.linalg.eig(W)
    w=e_vector[:,0]
    threshold=np.mean((w.T)*data,axis=1).tolist()[0]
    return w,threshold

def test(dataset,labelset,w,threshold):
    data=np.mat(dataset)
    label_test=(w.T)*data
    count=0
    for i in range(len(dataset)):
        flag=-1
        if labelset[i]>threshold[0]:
            flag=1
        else:
            flag=0
        if flag==labelset[i]:
            count+=1
    return count/len(dataset)

def run(dataset,dataset1,dataset0,labelset):
    w,threshold=train(dataset1,dataset0,dataset)
    rightrate=test(dataset,labelset,w,threshold)
    print("投影矩阵为：","\n",w)
    print("正确率：%.0f%%"%(rightrate*100))
    
dataset=[[0.697,0.774,0.634,0.608,0.556,0.403,0.481,0.437,0.666,0.243,0.245,0.343,0.639,0.657,0.360,0.593,0.719],
      [0.460,0.376,0.264,0.318,0.215,0.237,0.149,0.211,0.091,0.267,0.057,0.099,0.161,0.198,0.370,0.042,0.103]]
dataset1=[[0.697,0.774,0.634,0.608,0.556,0.403,0.481,0.437],
          [0.460,0.376,0.264,0.318,0.215,0.237,0.149,0.211]]
dataset0=[[0.666,0.243,0.245,0.343,0.639,0.657,0.360,0.593,0.719],
          [0.091,0.267,0.057,0.099,0.161,0.198,0.370,0.042,0.103]]
labelset=[1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0]
run(dataset,dataset1,dataset0,labelset)


