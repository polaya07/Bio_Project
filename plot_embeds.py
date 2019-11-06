import numpy as np
import pandas as pd
import os
import sys
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from deepcrispr import deepcrispr
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random
import glob

#data directory
data_dir = '../Ecoli_Training_dataset'
nn_savefile = 'savedmodels/deepcrispr.ckpt'

#load pam sites     
test_data = []
test_labels = []
files = list(glob.glob(os.path.join(data_dir,'*')))                               
for i,genome in enumerate(files[-5:]):                                       
    data = pd.read_csv(genome)
    test_data.extend(data['Seq+PAM'])
    test_labels.extend(data['Prediction'])
    
    sys.stdout.write("processing file %i       \r" % (i+1))
    sys.stdout.flush()

#select subset
idx = np.random.choice(np.arange(len(test_labels)),10000,replace=False)
test_data = [test_data[i] for i in idx]
test_labels = [test_labels[i] for i in idx]

#get embeddings
model = deepcrispr()
model.load(nn_savefile)
embeds = model.get_embeds(test_data)
    
#dimensionality reduction
#pca = PCA(n_components=2)
pca = TSNE(n_components=2)
embeds = pca.fit_transform(embeds)
'''
tsne = TSNE(n_components=2)
embeds = tsne.fit_transform(embeds)
'''
#plot embeddings
fig,ax = plt.subplots(figsize=(10,10))
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
im = ax.scatter(embeds[:,0],embeds[:,1],c=test_labels,s=3,alpha=0.5)
fig.colorbar(im, cax=cax, orientation='vertical')
plt.savefig('test.png',bbox_inches='tight',dpi=150)
plt.close()
