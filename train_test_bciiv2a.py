import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from visdom import Visdom
from model.LightConvNet import LightConvNet
from model.baseModel import baseModel
import time
import os
import yaml
from data.data_utils import *
from data.dataset import eegDataset

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

    for subId in range(1, 10):
        trainDataFiles = ['A0' + str(subId) + 'T']
        validationSet = config['validationSet']
        testDataFile = ['A0' + str(subId) + 'E']

        outPath = os.path.join(outFolder, config['network'], 'sub'+str(subId), randomFloder)
        
        if not os.path.exists(outPath):
            os.makedirs(outPath)
        
        print("Results will be saved in folder: " + outPath)

        dictToYaml(os.path.join(outPath, 'config.yaml'), config)

        setRandom(config['randomSeed'])

        data, labels = load_BCI42a_data(dataPath, trainDataFiles)
        trainData, trainLabels, valData, valLabels = split_data(data, labels, validationSet)
        testData, testLabels = load_BCI42a_data(dataPath, testDataFile)

        trainDataset = eegDataset(trainData, trainLabels)
        valDataset = eegDataset(valData, valLabels)
        testDataset = eegDataset(testData, testLabels)

        # vis = Visdom()

        netArgs = config['networkArgs']
        net = eval(config['network'])(**netArgs)
        print('Trainable Parameters in the network are: ' + str(count_parameters(net)))

        lossFunc = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=lrFactor, patience=lrPatience, min_lr=lrMin)

        model = baseModel(net, config, resultSavePath=outPath)

        model.train(trainDataset, valDataset, lossFunc, optimizer, scheduler=scheduler)

        classes = ['left hand', 'right hand', 'foot', 'tongue']
        model.test(testDataset, classes)

        # vis.close()

if __name__ == '__main__':
    configFile = 'config/bciiv2a_config.yaml'
    file = open(configFile, 'r', encoding='utf-8')
    config = yaml.full_load(file)
    file.close()
    main(config)

