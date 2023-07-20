import copy
import numpy as np
from scipy import rand
import torch
import torch.nn as nn
import torch.optim as optim
from visdom import Visdom
from model.LightConvNet import LightConvNet
from model.baseModel import baseModel
import time
import os
import yaml
from data.data_utils import *
from data.dataset import eegDataset
from sklearn.model_selection import StratifiedKFold

def setRandom(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dictToYaml(filePath, dictToWrite):
    with open(filePath, 'w', encoding='utf-8') as f:
        yaml.dump(dictToWrite, f, allow_unicode=True)
    f.close()

def generateBalancedFolds(idx, label, kFold=10):
    folds = []
    skf = StratifiedKFold(n_splits=kFold)
    for train_index, test_index in skf.split(idx, label):
        folds.append([idx[i] for i in test_index])
    return folds

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(config):
    dataPath = config['dataPath']
    outFolder = config['outFolder']
    randomFloder = str(time.strftime('%Y-%m-%d--%H-%M', time.localtime()))

    lr = config['lr']
    lrFactor = config['lrFactor']
    lrPatience = config['lrPatience']
    lrMin = config['lrMin']

    for subId in range(1,55):
        dataFile = str(subId).zfill(2) + 'T.mat'

        outPath = os.path.join(outFolder, 'cv', config['network'], 'sub'+str(subId), randomFloder)
        
        if not os.path.exists(outPath):
            os.makedirs(outPath)
        
        print("Results will be saved in folder: " + outPath)

        dictToYaml(os.path.join(outPath, 'config.yaml'), config)

        setRandom(config['randomSeed'])

        netArgs = config['networkArgs']
        init_net = eval(config['network'])(**netArgs)
        print('Trainable Parameters in the network are: ' + str(count_parameters(init_net)))

        lossFunc = nn.CrossEntropyLoss()

        data, labels = load_openBMI_data(os.path.join(dataPath, dataFile))

        subIdx = list(range(data.shape[0]))
        subIdxFolds = generateBalancedFolds(subIdx, labels, kFold=config['kFold'])
        
        classes = ['left hand', 'right hand']

        testResultsCV = []

        for i, fold in enumerate(subIdxFolds):
            testIdx = fold
            tempFolds = copy.deepcopy(subIdxFolds)
            if i+1 < config['kFold']:
                valIdx = tempFolds[i+1]
            else:
                valIdx = tempFolds[0]
            tempFolds.remove(testIdx)
            tempFolds.remove(valIdx)
            trainIdx = [idx for f in tempFolds for idx in f]

            trainData = np.array([data[i] for i in trainIdx])
            trainLabels = np.array([labels[i] for i in trainIdx])
            valData = np.array([data[i] for i in valIdx])
            valLabels = np.array([labels[i] for i in valIdx])
            testData = np.array([data[i] for i in testIdx])
            testLabels = np.array([labels[i] for i in testIdx])

            trainDataset = eegDataset(trainData, trainLabels)
            valDataset = eegDataset(valData, valLabels)
            testDataset = eegDataset(testData, testLabels)

            outFold = os.path.join(outPath, 'fold'+str(i))
            os.makedirs(outFold)

            # vis = Visdom()

            net = copy.deepcopy(init_net)
            optimizer = optim.Adam(net.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lrFactor, 
                                                            patience=lrPatience, min_lr=lrMin)

            model = baseModel(net, config, crossValidation=True, resultSavePath=outFold)
            model.train(trainDataset, valDataset, lossFunc, optimizer, scheduler=scheduler)

            testResult = model.test(testDataset, classes)
            testResultsCV.append(testResult)

            # vis.close()

        testAcc = np.array([r['acc'] for r in testResultsCV])
        testCm = np.array([r['cm'] for r in testResultsCV])

        avgAcc = np.mean(testAcc)
        avgCm = np.mean(testCm, axis=0)

        avgResult = {'acc': avgAcc, 'cm': avgCm}

        print('Average accuracy:', avgAcc)
        with open(os.path.join(outPath, 'TestAvgResult.txt'), 'w') as fp:
            for key, value in avgResult.items():
                fp.write(f'{key}: {value}\n'.format(key, value))

if __name__ == '__main__':
    configFile = 'config/openbmi_cv_config.yaml'
    file = open(configFile, 'r', encoding='utf-8')
    config = yaml.full_load(file)
    file.close()
    main(config)
