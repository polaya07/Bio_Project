import tensorflow as tf
from deepcrispr import DCModelOntar
import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd

#Read all sequences from files
def readseqs(path):
    data = pd.read_csv(join(path,onlyfiles[i]), usecols=[0,1], sep="	", header=0)
    data['Seq+PAM']=data['Seq']+data['PAM']
    seqs=data['Seq+PAM'].tolist()
    return seqs, data 

def str_to_numpy(batch):

    bp2idx = {'A':0,'C':1,'G':2,'T':3}
    batch_size = len(batch)
    array = np.zeros((batch_size,4,1,23))
    print (array.shape)
    for i,seq in enumerate(batch):
        for j,bp in enumerate(seq):
            if bp not in bp2idx:
                continue
            p = bp2idx[bp]
            array[i,p,0,j] = 1
    return array

#Model from DeepCRISPR

#Flags from DeepCRISPR model
seq_feature_only = True
channels = 4 if seq_feature_only else 8

sess = tf.InteractiveSession()
on_target_model_dir = './ontar_cnn_reg_seq/'
# using regression model, otherwise classification model
is_reg = True

# using sequences feature only, otherwise sequences feature + selected epigenetic features
seq_feature_only = True
dcmodel = DCModelOntar(sess, on_target_model_dir, is_reg, seq_feature_only)


#Read seqs from file
path = '/home/polaya/Downloads/Bioinformatics_Project/562_PAMsites' 
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
print ('Number of files: ', len(onlyfiles))
folds = 5
skipped =[]
for i in range(62, len(onlyfiles)):
        print ('File', i, ' ', onlyfiles[i])
        seqs, df = readseqs(path)
        print (len(seqs), 'sequences were read from file')
        if (len(seqs)>500000):
            print (onlyfiles[i], ' skipped')
            skipped.append(onlyfiles[i]) 
            continue
        #change from nucleotides to input format from DeepCRISPR
        x_on_target = str_to_numpy(seqs)     # [batch_size, channels, 1, 23]
        predicted_on_target = [0] * len(seqs)
        for j in range (0,folds):
            #print ('Iteration', j, predicted_on_target)
            predicted_on_target = predicted_on_target + dcmodel.ontar_predict(x_on_target)
        #print (predicted_on_target)
        df['Prediction']=predicted_on_target/5
        df['Label']= df['Prediction']>0.5
        df.to_csv('/home/polaya/Downloads/Bioinformatics_Project/trainingset/'+onlyfiles[i].split('.')[0]+'.'+onlyfiles[i].split('.')[1])
