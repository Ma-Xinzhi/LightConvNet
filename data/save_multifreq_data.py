import numpy as np
import os
from data_utils import *

# Todo adjust the save path
data_path = 'dataset/bci_iv_2a/raw'
train_data_files = ['A0'+str(i)+'T_data.npy' for i in range(1,10)]
train_label_files = ['A0'+str(i)+'T_label.npy' for i in range(1,10)]
test_data_files = ['A0'+str(i)+'E_data.npy' for i in range(1,10)]
test_label_files = ['A0'+str(i)+'E_label.npy' for i in range(1,10)]

save_path = 'dataset/bci_iv_2a/multifreq'

if not os.path.exists(save_path):
    os.makedirs(save_path)

filtBank = [[4,8],[8,12],[12,16],[16,20],[20,24],[24,28],[28,32],[32,36],[36,40]]
transform = filterBank(filtBank, 250)
numfiltBank = 9

for i in range(9):
    print(f'Processing subject No. {i+1}'.format(i))
    train_data = np.load(os.path.join(data_path, train_data_files[i]))
    train_label = np.load(os.path.join(data_path, train_label_files[i]))
    test_data = np.load(os.path.join(data_path, test_data_files[i]))
    test_label = np.load(os.path.join(data_path, test_label_files[i]))
    multifreq_train_data = np.zeros([train_data.shape[0], numfiltBank, *train_data.shape[1:3]])
    multifreq_test_data = np.zeros([test_data.shape[0], numfiltBank, *test_data.shape[1:3]])
    for j in range(train_data.shape[0]):
        multifreq_train_data[j,:,:,:] = transform(train_data[j])
        multifreq_test_data[j,:,:,:] = transform(test_data[j])
    np.save(os.path.join(save_path, 'A0'+str(i+1)+'T_data.npy'), multifreq_train_data)
    np.save(os.path.join(save_path, 'A0'+str(i+1)+'T_label.npy'), train_label)
    np.save(os.path.join(save_path, 'A0'+str(i+1)+'E_data.npy'), multifreq_test_data)
    np.save(os.path.join(save_path, 'A0'+str(i+1)+'E_label.npy'), test_label)

