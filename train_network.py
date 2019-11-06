import pandas as pd
import numpy as np
import glob
import os
import sys
import random
from deepcrispr import deepcrispr

#data directory
data_dir = '../Ecoli_Training_dataset'

#PAM strings
train_data = []
test_data = []
train_labels = []
test_labels = []

#load files
files = list(glob.glob(os.path.join(data_dir,'*')))

for i,genome in enumerate(files[:-5]):                                       
    data = pd.read_csv(genome)
    train_data.extend(data['Seq+PAM'])
    train_labels.extend(data['Prediction'])
    
    sys.stdout.write("processing file %i       \r" % (i+1))
    sys.stdout.flush()

for i,genome in enumerate(files[-5:]):                                       
    data = pd.read_csv(genome)
    test_data.extend(data['Seq+PAM'])
    test_labels.extend(data['Prediction'])
    
    sys.stdout.write("processing file %i       \r" % (i+1))
    sys.stdout.flush()
       
print('\nsaved %i pam sites' % len(data))
    
#train val split
print('shuffling data into train/val splits')
num_samples = len(train_data)
val_size = int(0.2 * num_samples)
#random.shuffle(data)
train_data = train_data[:-val_size]
val_data = train_data[-val_size:]
train_labels = train_labels[:-val_size]
val_labels = train_labels[-val_size:]

#train model
if not os.path.exists('savedmodels'):
    os.makedirs('savedmodels')

model = deepcrispr()
#model.pretrain(train_data,X_val=val_data,savepath='savedmodels/deepcrispr.ckpt')
model.train(train_data,train_labels,val_data,val_labels,savepath='savedmodels/deepcrispr.ckpt')
