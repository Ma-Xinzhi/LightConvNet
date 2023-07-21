import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import os
import copy
import itertools
from mpl_toolkits.axes_grid1 import host_subplot
from datetime import datetime

class baseModel():
    def __init__(self, net, config, crossValidation=False, resultSavePath=None, vis=None):
        self.crossValidation = crossValidation
        self.resultSavePath = resultSavePath
        
        self.batchsize = config['batchSize']
        self.preferedDevice = config['preferedDevice']
        self.vis = vis

        self.device = None
        self.setDevice(config['nGPU'])
        self.net = net.to(self.device)

        # for training
        self.optimizer = None
        self.lossFunc = None
        self.scheduler = None

        self.maxEpochs = config['maxEpochs']
        self.numEpochsEarlyStop = config['numEpochsEarlyStop']
        self.maxEpochsAfterEarlyStop = config['maxEpochsAfterEarlyStop']
        self.continueAfterEarlyStop = config['continueAfterEarlyStop']

    def setDevice(self, nGPU):
        if self.device is None:
            if self.preferedDevice == 'gpu':
                self.device = torch.device('cuda:'+str(nGPU) if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device('cpu')
            print("Code will be running on device ", self.device)

    def train(self, trainDataset, valDataset, lossFunc, optimizer, scheduler=None):
        self.lossFunc = lossFunc
        self.optimizer = optimizer
        self.scheduler = scheduler

        trainResults = self._train(trainDataset, valDataset)
    
        if not self.crossValidation and self.resultSavePath is not None:
            # with open(os.path.join(self.resultSavePath, 'TrainResults.pkl'), 'wb') as fp:
            #     pickle.dump(trainResults, fp)
            
            self.plotLossAndAcc(trainResults, savePath=os.path.join(self.resultSavePath, 'loss-acc.png'))

            torch.save(self.net.state_dict(), os.path.join(self.resultSavePath, 'model.pth'))

    def test(self, testDataset, classes):
        preds, trues = self.predict(testDataset)
        testResults = self.calculateResults(preds, trues, confusionMatrix=True, classes=classes)
        print('Test Accuracy Result: ', testResults['acc'])
        print('Test F1-score Result: ', testResults['f1-score'])
        
        if self.resultSavePath is not None:
            with open(os.path.join(self.resultSavePath, 'TestResults.txt'), 'w') as fp:
                for key, value in testResults.items():
                    fp.write(f'{key}: {value}\n'.format(key, value))
            if not self.crossValidation:
                saveCmPath = os.path.join(self.resultSavePath, 'Confusion Matrix.png')
                self.saveConfusionMatrix(testResults['cm'], classes, saveCmPath)

        return testResults

    def _train(self, trainDataset, valDataset):
        valResults = []
        trainLoss = []
        trainLossAfterEarlyStop = []

        bestNet = None
        bestOptimizerState = None

        epoch = 0
        bestValAcc = 0

        countEpoch = 0

        earlyStopReached = False
        doStop = False

        while not doStop:
            if not earlyStopReached:
                lossOneEpoch = self._trainOneEpoch(trainDataset)
                trainLoss.append(lossOneEpoch)
                if self.scheduler is not None:
                    self.scheduler.step(lossOneEpoch)

                pred, act = self.predict(valDataset)
                valAcc = self.calculateResults(pred, act)['acc']
                valResults.append(valAcc)

                if self.vis is not None:
                    self.vis.line([lossOneEpoch], [epoch], win='Loss vs Accuracy', name='Loss', update='append')
                    self.vis.line([valAcc], [epoch], win='Loss vs Accuracy', name='Acc', update='append')

                print('Epoch [%d] | Train Loss: %.4f | Val Acc: %.4f | lr: %.6f' 
                      %(epoch+1, lossOneEpoch, valAcc, self.optimizer.param_groups[0]['lr']))

                if bestValAcc <= valAcc:
                    bestValAcc = valAcc
                    countEpoch = 0
                    bestNet = copy.deepcopy(self.net.state_dict())
                    bestOptimizerState = copy.deepcopy(self.optimizer.state_dict())
                else:
                    countEpoch += 1
                
                epoch += 1
                
                if (countEpoch >= self.numEpochsEarlyStop) or (epoch >= self.maxEpochs):
                    doStop = True

                    self.net.load_state_dict(bestNet)
                    self.optimizer.load_state_dict(bestOptimizerState)
                    
                    if self.continueAfterEarlyStop:
                        doStop = False
                        earlyStopReached = True
                        print('Early stop reached now continuing with full set')
                        trainDataset.combineDataset(valDataset)
                        epoch = 0
                        if self.scheduler is not None:
                            self.scheduler._reset()
            else:
                lossOneEpoch = self._trainOneEpoch(trainDataset)
                trainLossAfterEarlyStop.append(lossOneEpoch)
                if self.scheduler is not None:
                    self.scheduler.step(lossOneEpoch)

                print('Epoch [%d] | Train Loss: %.4f | lr: %.6f' 
                      %(epoch+1, lossOneEpoch, self.optimizer.param_groups[0]['lr']))

                if self.vis is not None:
                    self.vis.line([lossOneEpoch], [epoch], win='Loss After Early Stop', update='append')
                
                epoch += 1

                if epoch >= self.maxEpochsAfterEarlyStop:
                    doStop = True
        
        return {'trainLoss': trainLoss, 'trainLossAfterEarlyStop': trainLossAfterEarlyStop, 
                'valAcc': valResults}

    def _trainOneEpoch(self, trainDataset):
        self.net.train()

        running_loss = 0

        dataLoader = DataLoader(trainDataset, batch_size=self.batchsize, shuffle=True)

        # start = datetime.now()

        with torch.enable_grad():
            for data, label in dataLoader:
                self.optimizer.zero_grad()

                data = data.type(torch.FloatTensor).to(self.device)
                label = label.type(torch.LongTensor).to(self.device)

                output = self.net(data)

                loss = self.lossFunc(output, label)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
        
        # end = datetime.now()

        # print("training time comsumption: {}ms".format((end-start).microseconds / 1000.0))
        
        return running_loss/len(dataLoader)

    def predict(self, dataset):
        predicted = []
        actual = []
        dataLoader = DataLoader(dataset, batch_size=self.batchsize)
        self.net.eval()

        # start = datetime.now()

        with torch.no_grad():
            for input, label in dataLoader:
                input = input.type(torch.FloatTensor).to(self.device)
                label = label.type(torch.LongTensor).to(self.device)

                output = self.net(input)
                preds = torch.max(output, 1)[1]

                predicted.extend(preds.tolist())
                actual.extend(label.tolist())
        
        # end = datetime.now()

        # print("prediction time comsumption: {}ms".format((end-start).microseconds / 1000.0))
        
        return predicted, actual

    def calculateResults(self, preds, trues, confusionMatrix=False, classes=None):
        acc = accuracy_score(trues, preds)

        if classes is None:
            if confusionMatrix:
                cm = confusion_matrix(trues, preds)
                return {'acc':acc, 'cm':cm}
            else:
                return {'acc':acc}
        else:
            f1score = f1_score(trues, preds, labels=range(len(classes)), average='weighted')
            if confusionMatrix:
                cm = confusion_matrix(trues, preds, labels=range(len(classes)), normalize='true')
                return {'acc':acc, 'f1-score':f1score, 'cm':cm}
            return {'acc':acc, 'f1-score':f1score}

    def plotLossAndAcc(self, trainResults, savePath=None):
        host = host_subplot(111)
        plt.subplots_adjust(right=0.8)
        part1 = host.twinx()

        host.set_xlabel('Training Epochs')
        host.set_ylabel('Training Loss')
        part1.set_ylabel('Validation Accuracy')

        host.plot(range(1, len(trainResults['trainLoss'])+1), trainResults['trainLoss'], label='Loss')
        part1.plot(range(1, len(trainResults['valAcc'])+1), trainResults['valAcc'], label='Acc')
        host.legend(loc=5)
        # plt.draw()

        if savePath is not None:
            plt.savefig(savePath)
        else:
            plt.show()

        plt.close()

    def saveConfusionMatrix(self, confusionMatrix, classes, savePath):
        cmap = plt.cm.Blues
        plt.figure(figsize=(14,12))
        plt.imshow(confusionMatrix, interpolation='nearest', cmap=cmap)
        plt.title('Confusion Matrix', fontsize=40)
        plt.colorbar()

        tickMarks = np.arange(len(classes))
        plt.xticks(tickMarks, classes, fontsize=20)
        plt.yticks(tickMarks, classes, fontsize=20)

        thresh = confusionMatrix.max() / 2.
        for i, j in itertools.product(range(confusionMatrix.shape[0]), range(confusionMatrix.shape[1])):
            plt.text(j, i, format(confusionMatrix[i, j], '.2f'),
                     horizontalalignment='center',
                     color='white' if confusionMatrix[i, j] > thresh else 'black',
                     fontsize=15)

        plt.ylabel('True Label', fontsize=30)
        plt.xlabel('Predicted Label', fontsize=30)

        assert savePath is not None
        plt.savefig(savePath)
        
        plt.close()



    