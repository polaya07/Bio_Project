import tensorflow as tf
import numpy as np
import sys

class deepcrispr(object):
   
    def __init__(self):
    
        self.bp2idx = {'A':0,'C':1,'G':2,'T':3}
        self.initializer = tf.contrib.layers.xavier_initializer()
    
        #inputs
        self.inputs = tf.placeholder(tf.float32,shape=[None,23,4])
        self.training = tf.placeholder(tf.bool)
    
        #encoder layers
        conv1 = tf.layers.conv1d(self.inputs,32,3,1,'same',
                activation=tf.nn.relu,kernel_initializer=self.initializer)
        bn_fw1 = tf.layers.batch_normalization(conv1,training=self.training)
        noise1 = tf.cond(self.training,
                 lambda:tf.add(bn_fw1,tf.random.normal(tf.shape(bn_fw1),0,0.1)),
                 lambda:bn_fw1)
        
        conv2 = tf.layers.conv1d(noise1,64,3,2,'same',
                activation=tf.nn.relu,kernel_initializer=self.initializer)    
        bn_fw2 = tf.layers.batch_normalization(conv2,training=self.training)      
        noise2 = tf.cond(self.training,
                 lambda:tf.add(bn_fw2,tf.random.normal(tf.shape(bn_fw2),0,0.1)),
                 lambda:bn_fw2)
        
        conv3 = tf.layers.conv1d(noise2,64,3,1,'same',
                activation=tf.nn.relu,kernel_initializer=self.initializer)    
        bn_fw3 = tf.layers.batch_normalization(conv3,training=self.training)      
        noise3 = tf.cond(self.training,
                 lambda:tf.add(bn_fw3,tf.random.normal(tf.shape(bn_fw3),0,0.1)),
                 lambda:bn_fw3)
        
        conv4 = tf.layers.conv1d(noise3,256,3,2,'same',
                activation=tf.nn.relu,kernel_initializer=self.initializer)    
        bn_fw4 = tf.layers.batch_normalization(conv4,training=self.training)      
        noise4 = tf.cond(self.training,
                 lambda:tf.add(bn_fw4,tf.random.normal(tf.shape(bn_fw4),0,0.1)),
                 lambda:bn_fw4)

        conv5 = tf.layers.conv1d(noise4,256,3,1,'same',
                activation=tf.nn.relu,kernel_initializer=self.initializer)    
        bn_fw5 = tf.layers.batch_normalization(conv5,training=self.training)      
        noise5 = tf.cond(self.training,
                 lambda:tf.add(bn_fw5,tf.random.normal(tf.shape(bn_fw5),0,0.1)),
                 lambda:bn_fw5)
        
        #latent embedding
        self.embed = tf.contrib.layers.flatten(noise5)
        reshape = tf.reshape(self.embed,(-1,6,1,256))
        
        #deconvolution layers
        deconv1 = tf.layers.conv2d_transpose(reshape,256,(3,1),(1,1),'same',
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
                p = self.bp2idx[bp]
                array[i,j,p] = 1
                
        return array
    
    def train(self,X,batch_size=128,iterations=500000):

        num_samples = len(X)
    
        for i in range(iterations):

            #select random sample
            batch_idx = np.random.randint(0,num_samples,batch_size)
            batch = [X[idx] for idx in batch_idx]
            batch = self._str_to_numpy(batch)

            feed_dict = {self.inputs:batch,self.training:True}
            loss,_ = self.sess.run([self.loss,self.optimizer],feed_dict=feed_dict)
            
            sys.stdout.write("iteration %i loss: %f       \r" % (i+1,loss))
            sys.stdout.flush()

    def score(self,X):
        pass
    
    def get_embeds(self,X):
        pass
    
    def save(self):
        self.saver.save(self.sess,filename)
    
    def load(self):
        self.saver.restore(self.sess,filename)
    
if __name__ == "__main__":

    #generate dummy data
    seqs = []
    for i in range(1000):
        seq = ''.join([np.random.choice(['A','C','G','T']) for i in range(23)])
        seqs.append(seq)

    #train model
    model = deepcrispr()
    model.train(seqs)