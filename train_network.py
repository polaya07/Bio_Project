import pandas as pd
import numpy as np
import glob
import os
import sys
import random
from deepcrispr import deepcrispr

'''
pretraining
'''

#data directory
data_dir = '../PAM_Sites'

#PAM strings
data = []

#load files
for i,genome in enumerate(glob.glob(os.path.join(data_dir,'*.txt'))):                                       
    with open(genome) as f:
        text = f.readlines()
    text = [l[:20]+l[21:24] for l in text[1:]]
    data.extend(text)
    
    sys.stdout.write("processing file %i       \r" % (i+1))
    sys.stdout.flush()
                
print('\nsaved %i pam sites' % len(data))
    
#train val split
print('shuffling data into train/val splits')
num_samples = len(data)
val_size = int(0.2 * num_samples)
#random.shuffle(data)
seqs = data[:-val_size]
val_seqs = data[-val_size:]

#train model
if not os.path.exists('savedmodels'):
    os.makedirs('savedmodels')

model = deepcrispr()
model.pretrain(seqs,X_val=val_seqs,savepath='savedmodels/deepcrispr.ckpt')

'''
training
'''

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
model.load('savedmodels/deepcrispr.ckpt')
model.train(train_data,train_labels,val_data,val_labels,savepath='savedmodels/deepcrispr.ckpt')
fscore,precision,recall = model.fscore(test_data,test_labels)
print('test set fscore: %.6f' % fscore)
print('test set precision: %.6f' % precision)
print('test set recall: %.6f' % recall)
