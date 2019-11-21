from sklearn.feature_extraction.text import CountVectorizer
import sys
import pandas as pd
from os import listdir
from os.path import isfile, join


def readseqs(path):
    seq_files = [f for f in listdir(path) if isfile(join(path, f))]
    for i in range(0, 5):#len(seq_files)):
        data = pd.read_csv(join(path,seq_files[i]), usecols=[3,4,5], header=0)
        target = data['Label']
        prob = data['Prediction']
        data['split']=(" ".join(data['Seq+PAM']))
        seqs= data['split'].tolist()
    return seqs, target, prob


path = sys.argv[1]
seqs, target, prob = readseqs(path)

#print ('Sequences in a list', seqs)
vectorizer = CountVectorizer(ngram_range=(3, 5))
X = vectorizer.fit_transform(seqs)
print(vectorizer.get_feature_names())
print(X.toarray())
