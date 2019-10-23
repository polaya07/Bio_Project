import numpy as np
import os
import sys
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from deepcrispr import deepcrispr
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import random

#input args
filename = '../PAM_Sites/562.1188.PATRIC.gff.guides.txt'
nn_savefile = 'savedmodels/deepcrispr.ckpt'

#load pam sites                                    
with open(filename) as f:
    text = f.readlines()
pamsites = [l[:20]+l[21:24] for l in text[1:]]

#select subset
random.shuffle(pamsites)
pamsites = pamsites[:30000]

#get embeddings
model = deepcrispr()
model.load(nn_savefile)
embeds = model.get_embeds(pamsites)
    
#dimensionality reduction
pca = PCA(n_components=2)
embeds = pca.fit_transform(embeds)
'''
tsne = TSNE(n_components=2)
embeds = tsne.fit_transform(embeds)
'''
#plot embeddings
fig,ax = plt.subplots(figsize=(10,10))
for i,embed in enumerate(embeds):
    ax.scatter(embed[0],embed[1],s=3,alpha=0.5,c='b')
plt.savefig('test.png',bbox_inches='tight',dpi=150)
plt.close()
