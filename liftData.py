import torch.utils.data as Data
import csv
import torch
import scipy.io as sio
import numpy as np


class LiftData(Data.Dataset):
    def __init__(self,dataPath,timeSeq=40,test=False,testScale = 0.2):
        if testScale < 0 or testScale > 1:
            raise ValueError("error testScale")
        self.isTest = test
        self.testScale = testScale

        mat_contents = sio.loadmat(dataPath)
        sampleData = mat_contents['sampleData'].squeeze()
        sampleLabel = mat_contents['sampleLabel'].squeeze()
        print("read matfile over!")

        trialData = []
        trialLabel = []
        subTrial = []
        last_label = -2
        bad_data = False
        for i, d in enumerate(sampleData):
            if sampleLabel[i] !=  last_label:
                subTrial = []
                last_label = sampleLabel[i]
                bad_data = True
            d =  d.astype(np.int)
            #print(np.max(d),np.min(d))
            if np.max(d)>4000 or np.min(d)<-4000:
                continue
            subTrial.append(d)
            if len(subTrial) and (len(subTrial) % timeSeq) == 0:
                if bad_data == True:
                    bad_data = False
                    subTrial = []
                    continue
                trialData.append(subTrial)
                trialLabel.append(sampleLabel[i])
                subTrial = []
        print("subTrial over!")
        self.train_data = torch.tensor(trialData,dtype = torch.float)#torch.tensor(trialData)
        self.train_labels = torch.tensor(trialLabel,dtype = torch.float)#torch.tensor(trialLabel,dtype=torch.long)
        self.totalLen = len(self.train_labels)
        self.testBeginIndex =  int(self.totalLen * (1 - self.testScale))

        self.test_data = self.train_data[self.testBeginIndex:]
        self.test_labels = self.train_labels[self.testBeginIndex:]

    def __getitem__(self, item):
        if self.isTest:
            realItem = item + self.testBeginIndex
        else:
            realItem = item
        eeg,label = self.train_data[realItem],self.train_labels[realItem]
        return eeg,label

    def __len__(self):
        if self.isTest:
            return self.totalLen - self.testBeginIndex
        else:
            return self.testBeginIndex

def inteSeriesData(pathList,writerPath):
    trialData = []
    trialLabels = []
    for (dataPath,eventsPath) in pathList:
        id_label = {}
        label = -1
        recording = False
        with open(eventsPath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # print(row['id'],row['HandStart'],row['FirstDigitTouch'],row['BothReleased'])
                if (row['HandStart'] == '1'):
                    recording = True
                    label = 1
                if (row['FirstDigitTouch'] == '1'):
                    recording = True  # todo 可以去掉
                    label = 0
                if (row['BothReleased'] == '1'):
                    recording = False
                if (recording):
                    id_label[row['id']] = label


        with open(dataPath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] in id_label:
                    singleTrial = [float(i) for i in row[1:]]
                    trialLabels.append(id_label[row[0]])
                    trialData.append(singleTrial)
    mat = {}
    mat['sampleData'] = trialData
    mat['sampleLabel'] = trialLabels
    sio.savemat(wp,mat)

if __name__=="__main__":

    dataPathList = []
    for i in range(1,9): #todo 当作传入参数
        dp = "D:/学习资料/data/Grasp-and-lift-EEG-challenge/train/train/subj1_series" + str(i) + "_data.csv"
        ep = "D:/学习资料/data/Grasp-and-lift-EEG-challenge/train/train/subj1_series" + str(i) + "_events.csv"
        wp = "D:/学习资料/data/Grasp-and-lift-EEG-challenge/train/train/subj1.mat"
        dataPathList.append((dp,ep))

    inteSeriesData(dataPathList,wp)




