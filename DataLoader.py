import numpy as np

import scipy.io as sio
import random


class DataLoader:

    def __init__(self, dataPath, batchSize, shaffle=True):

        self.m_dataPath = dataPath
        self.m_batchSize = batchSize
        self.m_shaffle = shaffle

        self.read_data(dataPath)

        self.dataSeq = []

        self.m_currentTrainSet = None
        self.m_currantBatch = 0


    def read_data(self, dataPath=None):
        path = dataPath or self.m_dataPath
        mat = sio.loadmat(path)

        self.m_Xtrain = mat["Yt"]
        self.m_Ytrain = mat["Ct"].astype(int)

        self.m_trainSize = self.m_Xtrain.shape[1]

        self.m_Xtest = mat["Yv"]
        self.m_Ytest = mat["Cv"].astype(int)

        self.m_testSize = self.m_Xtest.shape[1]

        self.reshuffle()

        x = 9

    def reshuffle(self):

        self.m_currentTrainSet = list(range(0, self.m_trainSize))

        if self.m_shaffle == True:
            random.shuffle(self.m_currentTrainSet)

        self.m_currentTestSet = list(range(0, self.m_testSize))

        self.m_currantBatch = 0

        x = 9

    def __iter__(self):

        # indexes = self.m_currentTrainSet[self.m_currantBatch:self.m_currantBatch+self.m_batchSize]

        # XtrainBatch = self.m_Xtrain[:,indexes]
        # YtrainBatch = self.m_Ytrain[:,indexes]

        # self.m_currantBatch = self.m_currantBatch+self.m_batchSize

        for i in np.arange(0, self.m_trainSize, self.m_batchSize):
            indexes = self.m_currentTrainSet[i:i + self.m_batchSize]

            yield (self.m_Xtrain[:, indexes], self.m_Ytrain[:, indexes])

    def get_test_set(self):

        return self.m_Xtest, self.m_Ytest
