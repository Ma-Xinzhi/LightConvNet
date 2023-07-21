import mne
from mne import event
from mne import epochs
from mne import label
import numpy as np
import scipy.io
import os

# Todo adjust the save path
data_path = 'E:/Dataset/BCI_Competition_IV/Datasets2a'
data_files = ['A0'+str(i)+'E.gdf' for i in range(1,10)]

label_path = 'E:/Dataset/BCI_Competition_IV/True_Labels/Datasets2a'
label_files = ['A0'+str(i)+'E.mat' for i in range(1,10)]

save_path = 'dataset/bci_iv_2a/raw'

if not os.path.exists(save_path):
    os.makedirs(save_path)

event_description = {'783':"CueUnknown"}

for file in data_files:
    raw_data = mne.io.read_raw_gdf(os.path.join(data_path, file), preload=True, verbose=False)

    raw_events, all_event_id = mne.events_from_annotations(raw_data)

    raw_data = mne.io.RawArray(raw_data.get_data()*1e6, raw_data.info)

    raw_data.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']

    test_picks = mne.pick_types(raw_data.info, eeg=True, exclude='bads')

    tmin, tmax = 0, 4

    # Unknown = 783
    event_id = dict()
    for event in all_event_id:
        if event in event_description:
            event_id[event] = all_event_id[event]

    raw_epochs = mne.Epochs(raw_data, raw_events, event_id, tmin, tmax, proj=True, picks=test_picks, baseline=None, preload=True)

    data = raw_epochs.get_data() # [n_epochs, n_channels, n_times]
    data = data[:, :, :-1]

    np.save(os.path.join(save_path, file[:-4]+'_data.npy'), data)

for file in label_files:
    true_label = scipy.io.loadmat(os.path.join(label_path, file))
    label = true_label['classlabel']
    np.save(os.path.join(save_path, file[:-4]+'_label.npy'), label)

