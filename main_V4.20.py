#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tkinter import N
import pandas as pd
df = pd.read_csv('/home/fly/coding/data/ec_lt_10.csv')


# In[2]:


df = df.dropna()


# In[3]:


# df.info()
one_hot = [ 'product', 'hardware','component']
sentences = ['sentences']
dev_id = ['dev_id']
label = ['label']
top_k = ['top_k']
False_k = ['false']
T_path = ['T_path']
F_path = ['F_path']





sentences_list = []
for i in range(len(df)):
    sentences_list.append(eval(df['sen_vecs'].iloc[i]))


# In[9]:


df['sentences'] = sentences_list




# In[12]:


import torch
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
import numpy as np
import torch
import re
import random
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[13]:


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False # 关闭卷积优化
    torch.backends.cudnn.deterministic = True # 使用确定性的操作，该语句在新版本torch中被以下语句替代
#     torch.use_deterministic_algorithms(False)
#     torch.backends.cudnn.enabled = False 
#     torch.backends.cudnn.benchmark = False
#     os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
#     os.environ['PYTHONHASHSEED'] = str(seed)
def worker_init(worked_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# In[14]:


set_seed(1)


# In[15]:


#创建子类  迭代器
class subDataset(Dataset.Dataset):
    #初始化，定义数据内容和标签
    def __init__(self, One_hot, Sentences, Dev_id, Label, Top_k, False_k, T_path, F_path):
        self.One_hot = pd.get_dummies(df[One_hot])
        self.Sentences = df[sentences]
        self.Dev_id = df[Dev_id]
        self.Label = df[Label]
        self.Topk = df[Top_k]
        self.Falsek = df[False_k]
        self.Tpath = df[T_path]
        self.Fpath = df[F_path]
        
    #返回数据集大小
    def __len__(self):
        return len(self.Sentences)
    #得到数据内容和标签
    def __getitem__(self, index):
        
        
        # dev_id = torch.Tensor(self.Dev_id.iloc[index].apply(lambda x:process_dev(x)).tolist()).squeeze().to(device)
        dev_id = list(self.Dev_id.iloc[index])
        
        top_k = self.Topk.iloc[index].tolist()
        
        false_k = self.Falsek.iloc[index].tolist()

        t_path = self.Tpath.iloc[index].tolist()

        f_path = self.Fpath.iloc[index].tolist()
        
        # one_hot = torch.Tensor(self.One_hot.iloc[index]).to(device)
        one_hot = torch.Tensor(self.One_hot.iloc[index]).to(device)
        label = torch.IntTensor(self.Label.iloc[index]).to(device)
        # print(label.size())
        # print(self.Sentences[index])
        sentences = torch.Tensor(self.Sentences.iloc[index]).squeeze().to(device)

        # one_hot_sentences = torch.cat((one_hot,sentences),0)[list(df_list['0'])[:128]].to(device)
        
        return  one_hot, sentences, dev_id, top_k, false_k, t_path, f_path, label


# In[16]:


dataset = subDataset(one_hot,sentences, dev_id, label,top_k, False_k, T_path, F_path)


# In[17]:


#创建DataLoader迭代器
dataloader = DataLoader.DataLoader(dataset,batch_size= 256, shuffle = True, num_workers= 0, worker_init_fn = worker_init(1))
# dataloader = DataLoader.DataLoader(dataset,batch_size= 128, num_workers= 0)





# In[28]:主网络


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# torch.cuda.manual_seed_all(1) #torch gpu下设置随机种子 （包括所有的gpu）
# torch.backends.cudnn.deterministic = True  # 保证每次结果一样

class MyNet(nn.Module):                    #继承来着nn.Module的父类
    def __init__(self,a,b,c,d):                    # 初始化网络
        super(MyNet, self).__init__()      #super()继承父类的构造函数，多继承需用到super函数
        
        self.U = nn.Parameter(d) #每个人的向量
        # self.U = nn.Parameter(torch.ones([1195, 64],dtype=torch.float).to(device)) #每个人的向量
        self.W1 = nn.Parameter(torch.randn(64, 64).to(device)) 
        self.W2 = nn.Parameter(torch.randn(64, 64).to(device))
        
        # self.dropout = nn.Dropout(p=0.1)
        self.linear1 = nn.Linear(424,32,bias=True)
        self.linear2 = nn.Linear(376,64,bias=True)

        self.bias1 = nn.Parameter(torch.randn(1).to(device))
        self.bias2 = nn.Parameter(torch.randn(1).to(device))
        self.a = a
        self.b = b
        self.c = c

    def forward(self, l_one_hot,l_sentences, false_k, t_path_s, t_path_t, f_path_s, f_path_t, label):

        sentences_v = self.linear1(l_sentences)
        
        x = self.linear2(torch.cat((l_one_hot,sentences_v),1)).unsqueeze(2) #32*768*1
            
        U1 = torch.mm(label, self.U).unsqueeze(1) #32*1*768
        # print(U1.size())
        p1 = nn.Sigmoid()(torch.matmul(U1,x)).squeeze(2)   #P1
        # print(p1.size())
        U2 = torch.matmul(false_k, self.U)  #32*10*768
        # print(U2.size())
        p2 = torch.sum(nn.Sigmoid()(torch.matmul(U2,x)), dim=1)  #P2
        # print(p2.size())
        U3 = torch.matmul(t_path_s, self.U)   #32*?*768
        U4 = torch.matmul(t_path_t, self.U)   #32*?*768
        #  p3 = torch.sum(nn.Sigmoid()(torch.matmul(torch.matmul(U3,self.W1)+torch.matmul(x.transpose(1,2),self.W2),U1.transpose(1,2))),dim=1)
        p3 = torch.sum(nn.Sigmoid()(torch.matmul(torch.matmul(U3,self.W1)+torch.matmul(x.transpose(1,2),self.W2),U1.transpose(1,2))),dim=1)
        # print(p3.size())
        U5 = torch.matmul(f_path_s, self.U)   #32*?*768
        U6 = torch.matmul(f_path_t, self.U)   #32*?*768
        # p4 = torch.matmul(torch.matmul(U5,self.W1)+torch.matmul(x.transpose(1,2),self.W2).reshape([-1,1,768]),U6.reshape([-1,768,1]))
        p4 = torch.matmul(U5,self.W1)+torch.matmul(x.transpose(1,2),self.W2)
        p4 = torch.sum(nn.Sigmoid()(torch.matmul(p4.unsqueeze(2),U6.unsqueeze(3))),dim=1).squeeze(2) 
        # print(p4.size())
        p5_11 = torch.matmul(U2,self.W1)
        p5_12 = torch.matmul(x.transpose(1,2).repeat(1,10,1),self.W2)
        # print(torch.matmul(U2,self.W1).size())
        # print(torch.matmul(x.transpose(1,2).repeat(1,10,1),self.W2).size())
        p5_13 = (p5_11+p5_12).unsqueeze(2)
        p5_14 = U1.repeat(1,10,1).unsqueeze(3)
        # print(p5_14.size())
        p5_1 = nn.Sigmoid()(torch.matmul(p5_13 ,p5_14)).squeeze(2)
        p5_2 = nn.Sigmoid()(torch.matmul(U2,x))
        p5 = torch.sum(torch.mul(p5_1,p5_2),dim=1)  # 按照位置相乘
        
        # p6 = -(p1-self.a*p2)-self.b*(p3-self.c*p4)+self.bias1*(p2-torch.ones([p2.size()[0],1]))+self.bias2(p1-p5)
        
        U7 = torch.cat((U1,U2,U3,U5,U6),1)  #32*?*768
        p7 = 0.01*torch.mean(torch.mean(torch.norm(U7,p=2,dim=2),dim=1).unsqueeze(1),dim=0)
        # print(p7.size())
        
        p8 = 0.01*torch.mean(torch.norm(x,p=2,dim=1),dim=0)
        # print(p8.size())
        
        p6 = -(p1-self.a*p2)-self.b*(p3-self.c*p4)+self.bias1*abs(p1+p2-torch.ones([p2.size()[0],1]).to(device))+self.bias2*abs(p1-p5)
        # p6 = torch.log(-(p1-self.a*p2)-self.b*(p3-self.c*p4)+self.bias1*abs(p1+p2-torch.ones([p2.size()[0],1]).to(device))+self.bias2*abs(p1-p5))
        
        # p6 = (p1-self.a*p2)+self.b*(p3-self.c*p4)-self.bias1*(p2-torch.ones([p2.size()[0],1]).to(device))-self.bias2*(p1-p5)
        
        # p6 = -torch.log(p1-self.a*p2)-self.b*torch.log(p3-self.c*p4)+self.bias1*torch.log(p1+p2-torch.ones([p2.size()[0],1]).to(device))+self.bias2*torch.log(p1-p5)
        
        return p6


# In[29]:


import torch.nn as nn
class myLoss(nn.Module):
    def __init__(self, device):
        super(myLoss, self).__init__()
        self.target = torch.FloatTensor([0]).to(device)
    def forward(self, outputs):
        return torch.mean(torch.pow((outputs - self.target), 2))
lossFunc=myLoss(device)


# In[30]:


from torch.optim.lr_scheduler import StepLR

mitri = torch.randn(1195, 64).to(device)

loss_function = myLoss(device)             # 定义损失函数为交叉熵损失函数 
ranknet_grad = nn.CrossEntropyLoss()




from torch.autograd import Variable
#训练过程

result_1 = pd.DataFrame()

a_list = []
b_list = []
top1 = []
top2 = []
top3 = []
top4 = []
top5 = []

m_list = [i for i in range(101)]
for m in m_list[1:]:
    m = m/100
    for n in m_list[1:]:
        n = n/100
        net = MyNet(n,m,n,mitri).to(device)
        optimizer = optim.Adam(net.parameters(), lr=0.01)# 定义优化器（训练参数，学习率）
        scheduler = StepLR(optimizer, step_size=2, gamma=0.0002)

        accuracys = []
        top_2_list = []
        top_3_list = []
        top_4_list = []
        top_5_list = []
        for i, item in enumerate(dataloader):
            print('i:', i)
            l_one_hot,l_sentences, l_dev_id,l_top_k,l_false_k, l_t_path, l_f_path, l_label = item
            ll_false_k = process_top_k(l_false_k) # 32*10*1195
            l_t_path_s, l_t_path_t = process_Path(l_t_path)  # 32*(?)*1195
            l_f_path_s, l_f_path_t = process_Path(l_f_path)
            ll_label = process_label(l_label)
            optimizer.zero_grad()# 历史梯度清0
            outputs = net(l_one_hot,l_sentences, ll_false_k.to(device), l_t_path_s.to(device), l_t_path_t.to(device), l_f_path_s.to(device), l_f_path_t.to(device), ll_label.to(device))
            #loss=Variable(outputs,requires_grad=True)
            loss1 = lossFunc(outputs)
            # x = torch.mm(l_one_hotsentences,net.linear)  #32*768
            sentences_v = net.linear1(l_sentences)
            # print(l_one_hot.size())
            x = net.linear2(torch.cat((l_one_hot,sentences_v),1))
            y = data_normal_2d(torch.matmul(x,net.U.T))
            # print(data_normal_2d(y))
            loss2 = ranknet_grad(y,ll_label.to(device))
            loss = 1*loss1 +0*loss2
            print("loss:============{}".format(loss))
            loss.backward()
            optimizer.step()
            scheduler.step()
            #     x = torch.mm(l_one_hotsentences,net.linear)  #32*768
            #     y = torch.matmul(x,net.U.T)
            # print(y.size())
            y_pre = y*processT_top_k(l_top_k,l_label.tolist()).to(device)
            
            # T_tok = processT_label_k(l_top_k,l_dev_id).to(device)
            # print(l_label.squeeze(1))
            accuracy = torch.eq(torch.max(y_pre,dim=1).indices, l_label.squeeze(1)).sum().item() / l_label.size(0)
            top_2 = TopK_accuracy(y_pre,l_label.squeeze(1),2)
            top_3 = TopK_accuracy(y_pre,l_label.squeeze(1),3)
            top_4 = TopK_accuracy(y_pre,l_label.squeeze(1),4)
            top_5 = TopK_accuracy(y_pre,l_label.squeeze(1),5)
            top_10 = TopK_accuracy(y_pre,l_label.squeeze(1),10)


            print("accuracy:============{}".format(accuracy))
            print("top_2:============{}".format(top_2))
            print("top_3:============{}".format(top_3))
            print("top_4:============{}".format(top_4))
            print("top_5:#############{}".format(top_5))
            print("top_10:#############{}".format(top_10))
            accuracys.append(accuracy)
            top_2_list.append(top_2)
            top_3_list.append(top_3)
            top_4_list.append(top_4)
            top_5_list.append(top_5)
        accuracys_mean = np.mean(accuracys)
        top_2_list_mean = np.mean(top_2_list)
        top_3_list_mean = np.mean(top_3_list)
        top_4_list_mean = np.mean(top_4_list)
        top_5_list_mean = np.mean(top_5_list)
        a_list.append(m)
        b_list.append(n)
        top1.append(accuracys_mean)
        top2.append(top_2_list_mean)
        top3.append(top_3_list_mean)
        top4.append(top_4_list_mean)
        top5.append(top_5_list_mean)

result_1['a'] = a_list
result_1['b'] = b_list
result_1['top1'] = top1
result_1['top2'] = top2
result_1['top3'] = top3
result_1['top4'] = top4
result_1['top5'] = top5
result_1.to_csv('result.csv',index=None)











# In[222]:


# max(accuracys[1:])


# In[233]:


# import matplotlib.pyplot as plt
# plt.figure(dpi=50,figsize=(32,10))
# x = np.array([i for i in range(len(accuracys[1:]))])
# y=np.array(accuracys[1:])

# plt.plot(x,y)




