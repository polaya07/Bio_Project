import numpy as np
import glob
import os
import sys
import random
from deepcrispr import deepcrispr

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
model.train(seqs,X_val=val_seqs,savepath='savedmodels/deepcrispr.ckpt')
