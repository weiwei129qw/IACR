
# In[24]:实验指标


def TopK_accuracy(output, target, topk):
    batch_size = target.size(0)
    y = target.tolist()
    pred = output.topk(topk,1, True, True).indices.tolist()
    flag = 0
    for i in range(len(y)):
        if y[i] in pred[i]:
            flag+=1
    return flag/batch_size
