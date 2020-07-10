import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def train(dataset,labelset,step,N):
    data=np.mat(dataset)    #创建data矩阵
    label=np.mat(labelset).transpose()  #创建label矩阵
    w = np.ones((len(dataset[0])+1, 1)) #为w矩阵多加一行以保存偏移参数b
    data = np.c_[data, np.ones((len(dataset), 1))]  #为data矩阵多加全1列，用于与偏移参数b相乘
    for i in range(N):
        label_pred=sigmoid(np.dot(data,w))
        label_delta=label_pred-label
        label_change=np.dot(data.transpose(),label_delta)
        w=w-label_change*step
    return w,step,N

def test(dataset,labelset,w):
    data=np.mat(dataset)
    a = np.ones((len(dataset), 1))
    data = np.c_[data, a]
    count=0
    label_test=sigmoid(np.dot(data,w))
    for i in range(len(labelset)):
        flag=-1
        if label_test[i]>0.5:
            flag=1
        else:
            flag=0
        if labelset[i]==flag:
            count+=1
    return count,count/len(dataset)

def run(dataset,labelset):
    w,step,N=train(dataset,labelset,0.001,1000)
    count,righrrate=test(dataset,labelset,w)
    print("迭代次数为：%d"%(N))
    print("学习率步长为：%.3f"%(step))
    print("最终权重为：\n",w[0:2,:])
    print("最终偏移为：%f"%(w[2]))
    
dataset=[[0.697,0.460],[0.774,0.376],[0.634,0.264],[0.608,0.318],[0.556,0.215],[0.403,0.237],[0.481,0.149],[0.437,0.211],[0.666,0.091],[0.243,0.267],[0.245,0.057],[0.343,0.099],[0.639,0.161],[0.657,0.198],[0.360,0.370],[0.593,0.042],[0.719,0.103]]
labelset=[1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0]
run(dataset,labelset)
