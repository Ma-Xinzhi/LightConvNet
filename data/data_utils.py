import numpy as np
import random
import scipy.signal as signal
import scipy.io as io
import os
import resampy

def load_BCI42a_data(dataset_path, data_files):
    data = []
    label = []
    for i in range(len(data_files)):
        #Todo adjust the file name
        data_path = os.path.join(dataset_path, data_files[i] + '_data.npy')
        label_path = os.path.join(dataset_path, data_files[i] + '_label.npy')

        if i==0:
            data = np.load(data_path)
            label = np.load(label_path).squeeze()-1
        else:
            data_t = np.load(data_path)
            label_t = np.load(label_path).squeeze()-1
            data = np.concatenate((data, data_t), axis=0)
            label = np.concatenate((label, label_t), axis=0)
        
        print(data_files[i], 'load success')

    #Shuffle
    data, label = shuffle_data(data, label)

    print('Data shape: ', data.shape)
    print('Label shape: ', label.shape)

    return data, label

def load_openBMI_data(data_file):
    alldata = io.loadmat(data_file)
    data = alldata['data']
    label = alldata['labels'].squeeze()

    data, label = shuffle_data(data, label)

    print('Data shape: ', data.shape)
    print('Label shape: ', label.shape)

    return data, label

def shuffle_data(data, label):
    index = [i for i in range(len(data))]
    random.shuffle(index)
    shuffle_data = data[index]
    shuffle_label = label[index]
    return shuffle_data, shuffle_label

def split_data(data, labels, val_rate=0.2):
    data_size = data.shape[0]

    val_index = random.sample(range(data_size), int(data_size*val_rate))

    data_train, labels_train, data_val, labels_val = [], [], [], []
    for i in range(data_size):
        if i in val_index:
            data_val.append(data[i])
            labels_val.append(labels[i])
        else:
            data_train.append(data[i])
            labels_train.append(labels[i])

    data_train = np.array(data_train)
    labels_train = np.array(labels_train)
    data_val = np.array(data_val)
    labels_val = np.array(labels_val)

    print('Training data size: ', data_train.shape[0])
    print('Validation data size: ', data_val.shape[0])

    return data_train, labels_train, data_val, labels_val

def get_k_fold_data(k, fold, train_data, train_label):
    assert k > 1
    fold_size = train_data.shape[0] // k

    x_train, y_train = None, None
    for i in range(k):
        idx = slice(i*fold_size, (i+1)*fold_size)
        x_part, y_part = train_data[idx, :], train_label[idx]
        
        if i == fold:
            x_val, y_val = x_part, y_part
        elif x_train is None:
            x_train = x_part
            y_train = y_part
        else:
            x_train = np.concatenate((x_train, x_part), axis=0)
            y_train = np.concatenate((y_train, y_part), axis=0)
    
    return x_train, y_train, x_val, y_val

class selectTimeRange():
    def __init__(self, tmin, tmax, freq, axis):
        self.timeRange = range(int(tmin*freq), int(tmax*freq))
        self.axis = axis
    
    def __call__(self, data):
        output = np.take(data, self.timeRange, axis=self.axis)
        return output

class selectFreqAndTimeRange():
    def __init__(self, tmin, tmax, old_freq, new_freq, axis):
        self.old_freq = old_freq
        self.new_freq = new_freq
        self.timeRange = range(int(tmin*new_freq), int(tmax*new_freq))
        self.axis = axis
    
    def __call__(self, data):
        rs_data = resampy.resample(data, self.old_freq, self.new_freq, axis=self.axis)
        output = np.take(rs_data, self.timeRange, axis=self.axis)
        return output

class filterBank():
    def __init__(self, filtBank, freq, filtAllowance=2, axis=1, filtType='filter'):
        self.filtBank = filtBank
        self.freq = freq
        self.filtAllowance = filtAllowance
        self.axis = axis
        self.filtType = filtType

    def bandpassFilter(self, data, filtBand, freq, filtAllowance, axis, filtType):
        """
        Filter a signal using cheby2 iir filtering.

        Parameters
        ----------
        data: 2d/ 3d np array
            trial x channels x time
        filtBand: two element list containing the low and high cut off frequency in hertz.
            if any value is specified as None then only one sided filtering will be performed
        fs: sampling frequency
        filtAllowance: transition bandwidth in hertz
        filtType: string, available options are 'filtfilt' and 'filter'

        Returns
        -------
        dataOut: 2d/ 3d np array after filtering
            Data after applying bandpass filter.
        """
        aStop = 30 # stopband attenuation
        aPass = 3 # passband attenuation
        nFreq= freq/2.0 # Nyquist frequency
        
        if (filtBand[0] == 0 or filtBand[0] is None) and (filtBand[1] == None or filtBand[1] >= nFreq):
            # no filter
            print("Not doing any filtering. Invalid cut-off specifications")
            return data
        
        elif filtBand[0] == 0 or filtBand[0] is None:
            # low-pass filter
            print("Using lowpass filter since low cut hz is 0 or None")
            fPass =  filtBand[1]/ nFreq
            fStop =  (filtBand[1]+filtAllowance)/ nFreq
            # find the order
            [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
            b, a = signal.cheby2(N, aStop, fStop, 'lowpass')
        
        elif (filtBand[1] is None) or (filtBand[1] == nFreq):
            # high-pass filter
            print("Using highpass filter since high cut hz is None or nyquist freq")
            fPass =  filtBand[0]/ nFreq
            fStop =  (filtBand[0]-filtAllowance)/ nFreq
            # find the order
            [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
            b, a = signal.cheby2(N, aStop, fStop, 'highpass')
        
        else:
            # band-pass filter
            # print("Using bandpass filter")
            fPass =  (np.array(filtBand)/ nFreq).tolist()
            fStop =  [(filtBand[0]-filtAllowance)/ nFreq, (filtBand[1]+filtAllowance)/ nFreq]
            # find the order
            [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
            b, a = signal.cheby2(N, aStop, fStop, 'bandpass')

        if filtType == 'filtfilt':
            dataOut = signal.filtfilt(b, a, data, axis=axis)
        else:
            dataOut = signal.lfilter(b, a, data, axis=axis)
        return dataOut

    def __call__(self, data):
        out = np.zeros([len(self.filtBank), *data.shape])

        for i, filtBand in enumerate(self.filtBank):
            out[i, :, :] = self.bandpassFilter(data, filtBand, self.freq, self.filtAllowance, 
                                      self.axis, self.filtType)
        
        if len(self.filtBank) <= 1:
            out = np.squeeze(out, axis=0)

        return out


class timeFilterBank():
    def __init__(self, time_win, time_stride, filtBank, freq, time_start=0, time_end=1000,
                 filtAllowance=1, axis=1, filtType='filter'):
        self.time_win = time_win
        self.time_stride = time_stride
        self.filtBank = filtBank
        self.freq = freq
        self.time_start = time_start
        self.time_end = time_end
        self.filtAllowance = filtAllowance
        self.axis = axis
        self.filtType = filtType

    def timeSegment(self, data):
        data_size = self.time_end - self.time_start
        num_time = (data_size - self.time_win) // self.time_stride + 1
        time_segs = np.zeros([num_time, data.shape[0], self.time_win])
        for i in range(num_time):
            start = self.time_start + i*self.time_stride
            end = start + self.time_win
            time_segs[i] = data[:, start:end]
        
        return time_segs, num_time


    def bandpassFilter(self, data, filtBand, freq, filtAllowance, axis, filtType):
        """
        Filter a signal using cheby2 iir filtering.

        Parameters
        ----------
        data: 2d/ 3d np array
            trial x channels x time
        filtBand: two element list containing the low and high cut off frequency in hertz.
            if any value is specified as None then only one sided filtering will be performed
        fs: sampling frequency
        filtAllowance: transition bandwidth in hertz
        filtType: string, available options are 'filtfilt' and 'filter'

        Returns
        -------
        dataOut: 2d/ 3d np array after filtering
            Data after applying bandpass filter.
        """
        aStop = 30 # stopband attenuation
        aPass = 3 # passband attenuation
        nFreq= freq/2.0 # Nyquist frequency
        
        if (filtBand[0] == 0 or filtBand[0] is None) and (filtBand[1] == None or filtBand[1] >= nFreq):
            # no filter
            print("Not doing any filtering. Invalid cut-off specifications")
            return data
        
        elif filtBand[0] == 0 or filtBand[0] is None:
            # low-pass filter
            print("Using lowpass filter since low cut hz is 0 or None")
            fPass =  filtBand[1]/ nFreq
            fStop =  (filtBand[1]+filtAllowance)/ nFreq
            # find the order
            [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
            b, a = signal.cheby2(N, aStop, fStop, 'lowpass')
        
        elif (filtBand[1] is None) or (filtBand[1] == nFreq):
            # high-pass filter
            print("Using highpass filter since high cut hz is None or nyquist freq")
            fPass =  filtBand[0]/ nFreq
            fStop =  (filtBand[0]-filtAllowance)/ nFreq
            # find the order
            [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
            b, a = signal.cheby2(N, aStop, fStop, 'highpass')
        
        else:
            # band-pass filter
            # print("Using bandpass filter")
            fPass =  (np.array(filtBand)/ nFreq).tolist()
            fStop =  [(filtBand[0]-filtAllowance)/ nFreq, (filtBand[1]+filtAllowance)/ nFreq]
            # find the order
            [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
            b, a = signal.cheby2(N, aStop, fStop, 'bandpass')

        if filtType == 'filtfilt':
            dataOut = signal.filtfilt(b, a, data, axis=axis)
        else:
            dataOut = signal.lfilter(b, a, data, axis=axis)
        return dataOut

    def __call__(self, data):
        data_seg, num_time = self.timeSegment(data)
        out = np.zeros([num_time, len(self.filtBank), *data_seg.shape[1:]])

        for time in range(num_time):
            for i, filtBand in enumerate(self.filtBank):
                out[time, i, :, :] = self.bandpassFilter(data_seg[time], filtBand, self.freq, self.filtAllowance, 
                                                   self.axis, self.filtType)
        
        if len(self.filtBank) <= 1:
            out = np.squeeze(out, axis=1)

        return out