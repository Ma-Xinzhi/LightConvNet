import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import os
import resampy
from data_utils import *

def fetchKoreaDataFile(dataPath, epochWindow=[0,4], chans=None, downsampleFactor=None):
    alldata = io.loadmat(dataPath)

    data = np.concatenate((alldata['EEG_MI_train'][0,0]['smt'], alldata['EEG_MI_test'][0,0]['smt']), axis=1)
    labels = np.concatenate((alldata['EEG_MI_train'][0,0]['y_dec'].squeeze(), alldata['EEG_MI_test'][0,0]['y_dec'].squeeze())).astype(int)-1

    allchans = np.array([m.item() for m in alldata['EEG_MI_train'][0,0]['chan'].squeeze()])
    fs = alldata['EEG_MI_train'][0,0]['fs'].squeeze().item()

    del alldata

    if chans is not None:
        data = data[:, :, chans]
        allchans = allchans[np.array(chans)]
    
    if downsampleFactor is not None:
        dataNew = np.zeros((int(data.shape[0]/downsampleFactor), *data.shape[1:3]), np.float)
        for i in range(data.shape[2]):
            dataNew[:, :, i] = resampy.resample(data[:, :, i], fs, fs//downsampleFactor, axis=0)
        data = dataNew
        fs = fs // downsampleFactor
    
    if epochWindow != [0,4]:
        start = epochWindow[0] * fs
        end = epochWindow[1] * fs
        data = data[start:end, :, :]

    # change the data dimension: trials x channels x time
    data = np.transpose(data, axes=(1, 2, 0))
    
    return {'data': data, 'labels': labels, 'chans': allchans, 'fs': fs}

def preprocessKoreaDataset(datasetPath, savePath, epochWindow=[0,4], 
                     chans=[7,32,8,9,33,10,34,12,35,13,36,14,37,17,38,18,39,19,40,20],
                     downsampleFactor=4):
    subjects = list(range(54))
    
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    print('Processed raw data will be saved in folder', savePath)

    for sub in subjects:
        print(f'Processing subject No. {sub+1}'.format(sub))
        trainFile = os.path.join(datasetPath, 'Sess01', 'sess01_subj'+str(sub+1).zfill(2)+'_EEG_MI.mat')
        testFile = os.path.join(datasetPath, 'Sess02', 'sess02_subj'+str(sub+1).zfill(2)+'_EEG_MI.mat')
        
        assert (os.path.exists(trainFile) and os.path.exists(testFile)), 'Do not find data, check the data path...'
            
        trainData = fetchKoreaDataFile(trainFile, epochWindow=epochWindow, chans=chans, downsampleFactor=downsampleFactor)
        testData = fetchKoreaDataFile(testFile, epochWindow=epochWindow, chans=chans, downsampleFactor=downsampleFactor)

        io.savemat(os.path.join(savePath, str(sub+1).zfill(2)+'T.mat'), trainData)
        io.savemat(os.path.join(savePath, str(sub+1).zfill(2)+'E.mat'), testData)

def multifreqKoreaDataset(datasetPath, savePath, numFreq, transform):
    subjects = list(range(54))
    
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    print('Processed multi-view data will be saved in folder', savePath)

    for sub in subjects:
        print(f'Processing subject No. {sub+1}'.format(sub))
        trainFile = os.path.join(datasetPath, str(sub+1).zfill(2)+'T.mat')
        testFile = os.path.join(datasetPath, str(sub+1).zfill(2)+'E.mat')
        
        assert (os.path.exists(trainFile) and os.path.exists(testFile)), 'Do not find data, check the data path...'
            
        trainData = io.loadmat(trainFile)
        testData = io.loadmat(testFile)

        train_data = trainData['data']
        test_data = testData['data']

        multifreq_train_data = np.zeros([train_data.shape[0], numFreq, *train_data.shape[1:3]])
        multifreq_test_data = np.zeros([test_data.shape[0], numFreq, *test_data.shape[1:3]])

        for i in range(train_data.shape[0]):
            multifreq_train_data[i,:,:,:] = transform(train_data[i])
            multifreq_test_data[i,:,:,:] = transform(test_data[i])

        trainData['data'] = multifreq_train_data
        testData['data'] = multifreq_test_data

        io.savemat(os.path.join(savePath, str(sub+1).zfill(2)+'T.mat'), trainData)
        io.savemat(os.path.join(savePath, str(sub+1).zfill(2)+'E.mat'), testData)


if __name__ == '__main__':
    # Todo adjust the save path
    datasetPath = 'E:/Dataset/OpenBMI'
    raw_savePath = 'dataset/openBMI/raw'
    multifreq_savePath = 'dataset/openBMI/multifreq'

    preprocessKoreaDataset(datasetPath, raw_savePath)

    filtBank = [[4,8],[8,12],[12,16],[16,20],[20,24],[24,28],[28,32],[32,36],[36,40]]
    transform = filterBank(filtBank, 250)
    numFreq = 9

    multifreqKoreaDataset(raw_savePath, multifreq_savePath, numFreq, transform)