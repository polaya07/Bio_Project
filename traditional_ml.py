import pandas as pd
import numpy as np
import glob
import os
import sys
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Lasso,LogisticRegression
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.ensemble import RandomForestClassifier

#data directory
data_dir = '../Ecoli_Training_dataset'

#PAM strings
train_data = []
test_data = []
train_labels = []
test_labels = []

#load files
files = list(glob.glob(os.path.join(data_dir,'*')))

for i,genome in enumerate(files[:100]):                                       
    data = pd.read_csv(genome)
    train_data.extend(data['Seq+PAM'])
    train_labels.extend(data['Prediction'])
    
    sys.stdout.write("processing train file %i       \r" % (i+1))
    sys.stdout.flush()

for i,genome in enumerate(files[-30:]):                                       
    data = pd.read_csv(genome)
    test_data.extend(data['Seq+PAM'])
    test_labels.extend(data['Prediction'])
    
    sys.stdout.write("processing test file %i       \r" % (i+1))
    sys.stdout.flush()
    
#count vectorizer
print("vectorizing sequences")
vectorizer = CountVectorizer(analyzer='char',ngram_range=(3,5))
train_data = vectorizer.fit_transform(train_data)
test_data = vectorizer.transform(test_data)
train_labels_rounded = (np.array(train_labels) >= 0.5).astype(np.int32)
true = (np.array(test_labels) >= 0.5).astype(np.int32)

#linear regression
print("training linear regression")
lr = Lasso()
lr.fit(train_data,train_labels)
preds = lr.predict(test_data)

preds = (np.array(preds) >= 0.5).astype(np.int32)
fscore = f1_score(true,preds)
precision = precision_score(true,preds)
recall = recall_score(true,preds)
print('linear regression test set fscore: %.6f' % fscore)
print('linear regression test set precision: %.6f' % precision)
print('linear regression test set recall: %.6f' % recall)
'''
#logistic regression
lr = LogisticRegression(penalty='l1')
lr.fit(train_data,train_labels_rounded)
preds = lr.predict(test_data)

fscore = f1_score(true,preds)
precision = precision_score(true,preds)
recall = recall_score(true,preds)
print('logistic regression test set fscore: %.6f' % fscore)
print('logistic regression test set precision: %.6f' % precision)
print('logistic regression test set recall: %.6f' % recall)

#random forest
lr = RandomForestClassifier(n_estimators=128)
lr.fit(train_data,train_labels_rounded)
preds = lr.predict(test_data)

fscore = f1_score(true,preds)
precision = precision_score(true,preds)
recall = recall_score(true,preds)
print('random forest test set fscore: %.6f' % fscore)
print('random forest test set precision: %.6f' % precision)
print('random forest test set recall: %.6f' % recall)
'''
