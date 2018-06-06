import torch
from torch.utils.data import dataloader
import torch.nn as nn
from liftData import LiftData
from bcicivData import bcicivData
from torch.autograd import Variable
import matplotlib.pyplot as plt
import threading

EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
OUT_LEN = 1
BATCH_SIZE = 1
TIME_STEP =  40          # rnn time step / image height
INPUT_SIZE = 32#59#         # rnn input size / image width
HIDDEN_SIZE = 64
LR= 0.001
LR_GRU = 0.001              # learning rate
LR_linear = 0.001
DOWNLOAD_MNIST = False   # set to True if haven't download the data
NEED_SHUFFLE = True

DataPath = "D:/学习资料/data/Grasp-and-lift-EEG-challenge/train/train/subj1.mat"
matPath = "D://学习资料//data//BCICIV_1_mat//BCICIV_calib_ds1a.mat"



class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.GRU(  #GRUGRU
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(HIDDEN_SIZE, OUT_LEN)

    def forward(self, x):
        r_out, h = self.rnn(x, None)  # None represents zero initial hidden state

        # out = nn.functional.sigmoid(self.out(r_out[:, -1, :]))
        out = nn.functional.tanh(self.out(r_out[:, -1, :]))
        return out

class trainEEG():
    def __init__(self):
        self.rnn = RNN()
        self.optimizer = optimizer = torch.optim.Adam(self.rnn.parameters(), lr=LR)   # optimize all cnn parameters#torch.optim.SGD([{"params":self.rnn.rnn.parameters(),'lr':LR_GRU},{"params":self.rnn.out.parameters(),'lr':LR_linear}],momentum=0.9)
        self.loss_func = nn.MSELoss(size_average=False)#nn.CrossEntropyLoss()
        self.train_data = LiftData(DataPath)#bcicivData(matPath)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_data, batch_size=BATCH_SIZE, shuffle=NEED_SHUFFLE)
        self.test_data = LiftData(DataPath,test=True)#bcicivData(matPath, test=True)
        self.test_x = torch.tensor(self.test_data.test_data)  # Variable(test_data.test_data, volatile=True).type(torch.FloatTensor)
        self.test_y = torch.tensor(self.test_data.test_labels)
        self.updateLock = threading.Lock()

        self.BLOCK = 400
    def testCurrentNN(self, test_x, test_y, epoch,step,loss):
        test_output = self.rnn(test_x)  # (samples, time_step, input_size)
        # pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
        #pred_y = torch.round(torch.max(test_output, 1)[0])
        pred_y = torch.tensor([-1 if i < 0 else 1 for i in  test_output.view(-1)])
        right_num = torch.sum(pred_y == torch.tensor(test_y,dtype = torch.long))
        #print("right_num", right_num)
        accuracy = float(right_num) / len(test_y)
        print('Epoch: ', epoch, 'step:', step, '| train loss: %.4f' % loss.item(), '| test accuracy: %.3f' % accuracy)

    def updateNet(self,netOutput,label):
        self.updateLock.acquire()
        self.loss_func(netOutput,label[0])
        self.updateLock.release()

    def trainOnline(self,eegQueue,label):
        eegList = []
        for ts in range(TIME_STEP):
            eegList.append(eegQueue.get())
        train_x = torch.tensor(eegList).view(BATCH_SIZE,TIME_STEP,INPUT_SIZE)
        #TODO 如果需要计算梯度更新权重 必须异步执行 if label==1 不更新网络 只输出结果


    def train(self):
        #return output [-1,1]
        # for frame in range(TIME_STEP):
        #     pass#todo 每帧训练一轮好还是TIME_STEP帧放在一起训练好？put进queue的时候直接将40帧当作一个数据放入？
        last_loss = 10
        all_loss = []
        for epoch in range(EPOCH):
            for step, (x, y) in enumerate(self.train_loader):  # gives batch data
                b_x = x  # Variable(x.view(-1, TIME_STEP, INPUT_SIZE))              # reshape x to (batch, time_step, input_size)
                b_y = y.view(BATCH_SIZE, -1)  # Variable(y)                               # batch y

                output = self.rnn(b_x)  # rnn output
                #print(output)
                # if abs(output-b_y)>0.5:
                #     continue
                loss = self.loss_func(output, b_y)  # cross entropy loss
                # if abs(last_loss - loss) < LR and self.optimizer.defaults['lr'] <= 0.1:
                #     self.optimizer.defaults['lr'] = self.optimizer.defaults['lr'] * 2
                last_loss = loss
                all_loss.append(loss)
                self.optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                self.optimizer.step()  # apply gradients

                if step % 5 == 0:
                    self.testCurrentNN(self.test_x, self.test_y,epoch,step,loss)
                    # print('current lr:',optimizer.defaults['lr'])
                    # print('grad:',rnn.rnn.weight_hh_l0.grad[10][:10])
                if (step + 1) % 100 == 0:
                    plt.figure()
                    plt.plot(all_loss)
                    plt.show()

if __name__ == '__main__':
    eegTrain = trainEEG()
    eegTrain.train()

'''
#rnn = nn.GRU(input_size=INPUT_SIZE,hidden_size=HIDDEN_SIZE,num_layers=1,batch_first=True)
rnn = RNN()
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
#optimizer = torch.optim.SGD([{"params":rnn.rnn.parameters(),'lr':LR_GRU},{"params":rnn.out.parameters(),'lr':LR_linear}],momentum=0.9)
loss_func = nn.BCEWithLogitsLoss(size_average = False)#torch.tensor([0.1,0.4])
#loss_func = nn.CrossEntropyLoss(torch.tensor([1.,5.]))                       # the target label is not one-hotted
all_loss = []

#train_data = LiftData(DataPath)
train_data = bcicivData(matPath)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=NEED_SHUFFLE)

#test_data = LiftData(DataPath,test=True)
test_data = bcicivData(matPath,test=True)
test_x = torch.tensor(test_data.test_data) #Variable(test_data.test_data, volatile=True).type(torch.FloatTensor)
test_y = torch.tensor(test_data.test_labels)


print(train_data.train_data.size())     # (60000, 28, 28)
print(train_data.train_labels.size())   # (60000)
print(train_data.__len__())
print(test_data.test_data.size())     # (60000, 28, 28)
print(test_data.test_labels.size())   # (60000)
print(test_data.__len__())

def testCurrentNN(test_x,test_y):
    test_output = rnn(test_x)  # (samples, time_step, input_size)
    #pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    pred_y =torch.round(torch.max(test_output,1)[0])
    right_num = torch.sum(pred_y == test_y)
    print("right_num",right_num)
    accuracy = float(right_num) / len(test_y)
    print('Epoch: ', epoch, 'step:', step, '| train loss: %.4f' % loss.item(), '| test accuracy: %.3f' % accuracy)

last_loss = 10
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):        # gives batch data
        b_x = x #Variable(x.view(-1, TIME_STEP, INPUT_SIZE))              # reshape x to (batch, time_step, input_size)
        b_y = y.view(BATCH_SIZE,-1) #Variable(y)                               # batch y

        output = rnn(b_x)                               # rnn output
        print(output)
        # if abs(output-b_y)>0.5:
        #     continue
        loss = loss_func(output, b_y)                   # cross entropy loss
        if abs(last_loss-loss) < LR and optimizer.defaults['lr'] <= 0.1:
            optimizer.defaults['lr'] = optimizer.defaults['lr'] * 2
        last_loss = loss
        all_loss.append(loss)
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients

        if step % 5 == 0:
            testCurrentNN(test_x,test_y)
            #print('current lr:',optimizer.defaults['lr'])
            #print('grad:',rnn.rnn.weight_hh_l0.grad[10][:10])
        if (step+1) % 100 == 0:
            plt.figure()
            plt.plot(all_loss)
            plt.show()

'''
# plt.figure()
# plt.plot(all_loss)
# plt.show()