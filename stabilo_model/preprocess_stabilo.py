'''
Preprocessing script for The OnHW Dataset: Online Handwriting Recognition from IMU-Enhanced Ballpoint Pens with Machine Learning.
'''

stable = True # True: latest version, False: stable version

import sys
ISCOLAB = 'google.colab' in sys.modules
if ISCOLAB:
    if stable: 
        !pip install tsai -q
    else:
        !pip install git+https://github.com/timeseriesAI/tsai.git -q
    
import tsai
from tsai.all import *
print('tsai       :', tsai.__version__)
print('fastai     :', fastai.__version__)
print('fastcore   :', fastcore.__version__)
print('torch      :', torch.__version__)
import zipfile
with zipfile.ZipFile("both_dep.zip", 'r') as zip_ref:
    zip_ref.extractall()

import pickle
import numpy as np
filename = 'both_dep.pkl'
infile = open(filename,'rb')
data= pickle.load(infile)
infile.close()

first_tuple = data[0:1]
first_tuple = np.array(first_tuple)

X_tr_temp = first_tuple[0, 0]
y_tr = first_tuple[0, 1]
X_test_temp = first_tuple[0, 2]
y_te = first_tuple[0, 3]

# fix shape of each sample from (timesteps, features) to (features, timesteps)
def fix_shape(data):
  num_samples = len(data)
  print("# of samples", num_samples)
  #data = torch.from_numpy(data)
  for i in range(num_samples):
    data[i] = torch.from_numpy(data[i])
    ts = data[i].shape[0]
    feats = data[i].shape[1] 
    data[i] = torch.transpose(data[i], 0, 1)
    # sanity checks
    assert data[i].shape[0] == feats
    assert data[i].shape[1] == ts

  return data

X_tr_temp = fix_shape(X_tr_temp)

print("fixed shape of first data sample:", X_tr_temp[0].shape)


def remove_bad_samples(data, labels, max_seq_len=500):
    num_samples = len(data)
    rm_indexes = [] # samples at these indexes have time steps > max_seq_len. Save them for removal.
    for i in range(num_samples):
      if(data[i].shape[1] > max_seq_len):
        rm_indexes.append(i)
    # TODO: clear rm_indexes list after all the bad samples have been removed.
    data = np.delete(data, rm_indexes)
    labels = np.delete(labels, rm_indexes)
    return data, labels


# find length (no. of timesteps) of longest sample in dataset
def max_seq_length(data):
  num_samples = len(data)
  max_seq_len = 0
  for i in range(num_samples):
    if(data[i].shape[1] > max_seq_len):
      max_seq_len = data[i].shape[1]
  return max_seq_len

X_tr_temp, y_tr = remove_bad_samples(X_tr_temp, y_tr, 40)
max_ts = max_seq_length(X_tr_temp)
print("[After removing bad samples] Length (# of timesteps) of longest multivariate sequence in dataset:", max_ts)

num_samples = len(X_tr_temp)
# create tensor to hold actual time step lengths of each training sample
X_tr_temp_len = torch.LongTensor(num_samples) 

for i in range(num_samples):
  X_tr_temp_len[i] = X_tr_temp[i].shape[1]

X_tr = np.zeros((num_samples, 13, max_ts))

# convert X_tr_temp to 3D
for i in range(num_samples):
  X_tr[i, :X_tr_temp[i].shape[0], :X_tr_temp[i].shape[1]] = X_tr_temp[i]

# now 3D array of shape (num_samples, num_features i.e. 13, num_timesteps i.e. max ts) is ready
X_tr = torch.from_numpy(X_tr)

'''
Pads batch of variable length
'''
global_mask = torch.tensor(X_tr != 0).to(device)
print(global_mask)

X_val = X_tr[0:2000, :, :]
y_val = y_tr[0:2000]
X_train = X_tr[2000:,:,:]
y_train = y_tr[2000:]
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

X, y, splits = combine_split_data([X_train, X_val], [y_train, y_val])

tfms  = [None, [Categorize()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=16, batch_tfms=[TSStandardize(by_var=True, verbose=True)], num_workers=0)
dls.show_batch(sharey=True)


