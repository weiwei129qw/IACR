

# In[18]:数据处理


def process_top_k(list_1):
    list_2 = []
    for i in list_1[0]:
        list_3 = []
        for j in eval(i):
            ids_list = [0]*1195
            ids_list[int(j)] = 1
            list_3.append(ids_list)
        list_2.append(list_3)
    #     print(torch.tensor(list_2).size())
    return torch.FloatTensor(list_2)  # 32*10*1195
# process_top_k(l_false_k)


# In[19]:数据处理


from torch.nn.utils.rnn import pad_sequence

def process_Path(list_1):
    list_source = []
    list_target = []
    for i in list_1[0]:
        list_2 = []
        list_3 = []
        for j in eval(i):
            ids_list = [0]*1195
            ids_list_1 = [0]*1195
            ids_list[int(j.split('*')[0])] = 1
            ids_list_1[int(j.split('*')[1])] = 1
            list_2.append(ids_list)
            list_3.append(ids_list_1)
        list_source.append(list_2)
        list_target.append(list_3)
    a = pad_sequence([torch.from_numpy(np.array(x)) for x in list_source], batch_first=True).float()
    b = pad_sequence([torch.from_numpy(np.array(x)) for x in list_target], batch_first=True).float()
    return a, b


# In[20]:


def process_label(num):
    list_1 = []
    for i in num.tolist():
        ids_list = [0]*1195
        ids_list[int(i[0])] = 1
        list_1.append(ids_list)
    return torch.FloatTensor(list_1)



def processT_top_k(list_1,list_label):
    list_2 = []
    # for i in range(len(list_1[0])):
    for i in range(len(list_1[0])):
        list_3 =  eval(list_1[0][i])[1:10]+list_label[i]
        ids_list = [0]*1195
        for j in list_3:
            ids_list[int(j)] = 1
        list_2.append(ids_list)
    #     print(torch.tensor(list_2).size())
    return torch.FloatTensor(list_2)  # 32*10*1195



# In[22]:数据处理


def processT_label_k(list_1,list_dev):
    list_2 = []
    # for i in range(len(list_1[0])):
    l_list_dev = list_dev[0]
    for i in range(len(list_1[0])):
        list_3 =  eval(list_1[0][i])+l_list_dev[i].split('*')
        list_3 = list_3[-10:]
        ids_list = [0]*1195
        for j in list_3:
            ids_list[int(j)] = 1
        list_2.append(ids_list)
    #     print(torch.tensor(list_2).size())
    return torch.FloatTensor(list_2)  # 32*10*1195


def data_normal_2d(orign_data,dim="col"):
    """
    针对于2维tensor归一化 数据处理
    可指定维度进行归一化，默认为行归一化
    参数1为原始tensor，参数2为默认指定行，输入其他任意则为列
    """
    if dim == "col":
        dim = 1
        d_min = torch.min(orign_data,dim=dim)[0]
        for idx,j in enumerate(d_min):
            if j < 0:
                orign_data[idx,:] += torch.abs(d_min[idx])
                d_min = torch.min(orign_data,dim=dim)[0]
    else:
        dim = 0
        d_min = torch.min(orign_data,dim=dim)[0]
        for idx,j in enumerate(d_min):
            if j < 0:
                orign_data[idx,:] += torch.abs(d_min[idx])
                d_min = torch.min(orign_data,dim=dim)[0]
    d_max = torch.max(orign_data,dim=dim)[0]
    dst = d_max - d_min
    if d_min.shape[0] == orign_data.shape[0]:
        d_min = d_min.unsqueeze(1)
        dst = dst.unsqueeze(1)
    else:
        d_min = d_min.unsqueeze(0)
        dst = dst.unsqueeze(0)
    norm_data = torch.sub(orign_data,d_min).true_divide(dst)
    return norm_data
