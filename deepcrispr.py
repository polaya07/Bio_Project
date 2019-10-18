#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="3"

import tensorflow as tf
import numpy as np
import sys
import random

class deepcrispr(object):
   
    def __init__(self):
    
        self.bp2idx = {'A':0,'C':1,'G':2,'T':3}
        self.initializer = tf.contrib.layers.xavier_initializer()
    
        #inputs
        self.inputs = tf.placeholder(tf.float32,shape=[None,23,4])
        self.training = tf.placeholder(tf.bool)
        self.dropout = tf.placeholder(tf.float32)
    
        #encoder layers
        conv1 = tf.layers.conv1d(self.inputs,32,3,1,'same',
                activation=tf.nn.relu,kernel_initializer=self.initializer)
        bn_fw1 = tf.layers.batch_normalization(conv1,training=self.training)
        drop1 = tf.nn.dropout(bn_fw1,self.dropout)
        
        conv2 = tf.layers.conv1d(drop1,64,3,2,'same',
                activation=tf.nn.relu,kernel_initializer=self.initializer)    
        bn_fw2 = tf.layers.batch_normalization(conv2,training=self.training)
        drop2 = tf.nn.dropout(bn_fw2,self.dropout)    
        
        conv3 = tf.layers.conv1d(drop2,64,3,1,'same',
                activation=tf.nn.relu,kernel_initializer=self.initializer)    
        bn_fw3 = tf.layers.batch_normalization(conv3,training=self.training)
        drop3 = tf.nn.dropout(bn_fw3,self.dropout)  
        
        conv4 = tf.layers.conv1d(drop3,128,3,2,'same',
                activation=tf.nn.relu,kernel_initializer=self.initializer)    
        bn_fw4 = tf.layers.batch_normalization(conv4,training=self.training)
        drop4 = tf.nn.dropout(bn_fw4,self.dropout)

        conv5 = tf.layers.conv1d(drop4,128,3,1,'same',
                activation=tf.nn.relu,kernel_initializer=self.initializer)    
        bn_fw5 = tf.layers.batch_normalization(conv5,training=self.training)
        drop5 = tf.nn.dropout(bn_fw5,self.dropout)
        
        #latent embedding
        self.embed = tf.contrib.layers.flatten(drop5)
        reshape = tf.reshape(self.embed,(-1,6,1,128))
        
        #deconvolution layers
        deconv1 = tf.layers.conv2d_transpose(reshape,128,(3,1),(1,1),'same',
                  activation=tf.nn.relu,kernel_initializer=self.initializer) 
        bn_bw1 = tf.layers.batch_normalization(deconv1,training=self.training)
        
        deconv2 = tf.layers.conv2d_transpose(bn_bw1,64,(3,1),(2,1),'same',
                  activation=tf.nn.relu,kernel_initializer=self.initializer) 
        bn_bw2 = tf.layers.batch_normalization(deconv2,training=self.training)

        deconv3 = tf.layers.conv2d_transpose(bn_bw2,64,(3,1),(1,1),'same',
                  activation=tf.nn.relu,kernel_initializer=self.initializer) 
        bn_bw3 = tf.layers.batch_normalization(deconv3,training=self.training)

        deconv4 = tf.layers.conv2d_transpose(bn_bw3,32,(3,1),(2,1),'same',
                  activation=tf.nn.relu,kernel_initializer=self.initializer) 
        bn_bw4 = tf.layers.batch_normalization(deconv4,training=self.training)

        self.reconst = tf.layers.conv2d_transpose(bn_bw4,4,(3,1),(1,1),'same',
                       activation=tf.nn.sigmoid,kernel_initializer=self.initializer) 
        self.reconst = tf.reshape(self.reconst[:,:-1,:,:],(-1,23,4))
        
        #loss and optimizer
        self.loss = tf.losses.mean_squared_error(self.inputs,self.reconst)
        self.optimizer = tf.train.AdamOptimizer(0.0001,0.9,0.99).minimize(self.loss)

        #init op
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())        
    
    def _str_to_numpy(self,batch):
    
        batch_size = len(batch)
        array = np.zeros((batch_size,23,4))
        for i,seq in enumerate(batch):
            for j,bp in enumerate(seq):
                if bp not in self.bp2idx:
                    continue
                p = self.bp2idx[bp]
                array[i,j,p] = 1
                
        return array
    
    def train(self,X,batch_size=128,epochs=50,patience=5,X_val=None,savepath=None):

        num_samples = len(X)
        print('training on %i samples' % num_samples)
        if not isinstance(X_val, type(None)):
            val_size = len(X_val)
            print('validating on %i samples' % val_size)
        shuffled = X.copy()
        pat_count = 0
        best_loss = np.inf
    
        #training epochs
        for ep in range(epochs):

            #shuffle
            print('shuffling data')
            random.shuffle(shuffled)

            train_loss = []

            #select batch
            for start in range(0,len(X),batch_size):
                
                if start+batch_size < len(X):
                    stop = start+batch_size
                else:
                    stop = len(X)
        
                batch = shuffled[start:stop]
                batch = self._str_to_numpy(batch)

                feed_dict = {self.inputs:batch,self.training:True,self.dropout:0.9}
                loss,_ = self.sess.run([self.loss,self.optimizer],feed_dict=feed_dict)
                train_loss.append(loss)
                
                sys.stdout.write("epoch %i sample %i loss: %f       \r" % (ep+1,stop,loss))
                sys.stdout.flush()

            print("epoch %i train loss: %f       \r" % (ep+1,np.mean(train_loss)))
            
            #check validation loss if available
            if not isinstance(X_val, type(None)):
                val_loss = self.score(X_val)
                print("epoch %i val loss: %f       \r" % (ep+1,val_loss))
                
                #save if performance better than previous best
                if val_loss < best_loss:
                    best_loss = val_loss
                    pat_count = 0
                    if savepath:
                        self.save(savepath)
                
                #check patience
                else:
                    pat_count += 1
                    if pat_count >= patience:
                        break

    def score(self,X,batch_size=128):
    
        total_loss = []
    
        for start in range(0,len(X),batch_size):
            
            if start+batch_size < len(X):
                stop = start+batch_size
            else:
                stop = len(X)
    
            batch = X[start:stop]
            batch = self._str_to_numpy(batch)
            feed_dict = {self.inputs:batch,self.training:False,self.dropout:1.0}
            loss = self.sess.run(self.loss,feed_dict=feed_dict)
            total_loss.append(loss)
            
            sys.stdout.write("validating sample %i     \r" % stop)
            sys.stdout.flush()
            
        print()
        return np.mean(total_loss)
    
    def get_embeds(self,X,batch_size=128):
    
        all_embeds = []
    
        for start in range(0,len(X),batch_size):
            
            if start+batch_size < len(X):
                stop = start+batch_size
            else:
                stop = len(X)
    
            batch = X[start:stop]
            batch = self._str_to_numpy(batch)
            feed_dict = {self.inputs:batch,self.training:False,self.dropout:1.0}
            embeds = self.sess.run(self.embed,feed_dict=feed_dict)
            all_embeds.append(embeds)
            
            sys.stdout.write("calculating sample %i     \r" % stop)
            sys.stdout.flush()
        
        print()
        all_embeds = np.concatenate(all_embeds,0)
        return all_embeds
        
    def save(self,filename):
        self.saver.save(self.sess,filename)
    
    def load(self,filename):
        self.saver.restore(self.sess,filename)
    
if __name__ == "__main__":

    #generate dummy data
    seqs = []
    for i in range(1000):
        seq = ''.join([np.random.choice(['A','C','G','T']) for i in range(23)])
        seqs.append(seq)
        
    val_seqs = []
    for i in range(1000):
        seq = ''.join([np.random.choice(['A','C','G','T']) for i in range(23)])
        val_seqs.append(seq)

    #train model
    model = deepcrispr()
    model.train(seqs,epochs=2,X_val=val_seqs)
    embeds = model.get_embeds(seqs)
    print(embeds.shape)
