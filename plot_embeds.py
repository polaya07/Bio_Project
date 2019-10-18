import numpy as np
import os
import sys
import random
from deepcrispr import deepcrispr

#input args
filename = '../PAM_Sites/562.1188.PATRIC.gff.guides.txt'
nn_savefile = 'savedmodels/deepcrispr.ckpt'

#load pam sites                                    
with open(filename) as f:
    text = f.readlines()
pamsites = [l[:20]+l[21:24] for l in text[1:]]

#get embeddings
model = deepcrispr()
model.load(nn_savefile)
embeds = model.get_embeds(pamsites)
    
#plot embeddings
