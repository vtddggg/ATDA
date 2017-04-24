from tensorlayer.layers import Conv2d,InputLayer,MaxPool2d,FlattenLayer,BatchNormLayer,DenseLayer,DropoutLayer
from tensorlayer.cost import cross_entropy
import tensorlayer as tl
import tensorflow as tf
import numpy as np
import random
import utils

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)

    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    for start_idx in range(0, len(inputs), batchsize):
        end_idx = start_idx + batchsize
        if end_idx > len(inputs):
            end_idx = start_idx + (len(inputs) % batchsize)

        if shuffle:
            excerpt = indices[start_idx:end_idx]

        else:
            excerpt = slice(start_idx, end_idx)

        yield inputs[excerpt], targets[excerpt]


def sample_candidatas(dataset,labels,candidates_num,shuffle=True):
    if shuffle:
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)
    excerpt = indices[0:candidates_num]
    candidates = dataset[excerpt]
    true_label = labels[excerpt]
    return candidates,true_label

def labeling_fuc(data,output1, output2, true_label, threshold=0.9):
    id = np.equal(np.argmax(output1,1),np.argmax(output2,1))
    output1 = output1[id,:]
    output2 = output2[id, :]
    data = data[id, :]
    true_label = true_label[id, :]
    max1 = np.max(output1,1)
    max2 = np.max(output2,1)
    id2=np.max(np.vstack((max1,max2)),0)>threshold
    output1 = output1[id2,:]
    data = data[id2, :]
    pseudo_label =utils.dense_to_one_hot(np.argmax(output1,1),10)
    true_label = true_label[id2, :]
    return data,pseudo_label,true_label


class ATDA(object):
    def __init__(self, sess,name='mnist-mnistm'):
        self.name=name
        self.sess=sess

    def create_model(self):

        if self.name == 'mnist-mnistm':
            drop_prob={'Ft':0.2, 'F1':0.5, 'F2':0.5}
            self.x = tf.placeholder(tf.float32, shape=[None, 28, 28, 3])
            self.y_ = tf.placeholder(tf.float32, shape=[None,10])
            self.istrain = tf.placeholder(tf.bool, shape=[])

            _input = InputLayer(self.x,name='input_layer')
            _shared_net = _input
            _shared_net=Conv2d(_shared_net, n_filter=32, filter_size=(5, 5), strides=(1, 1),
                               act=tf.nn.relu, padding='SAME', name='cnn1')
            _shared_net = MaxPool2d(_shared_net, filter_size=(2, 2), strides=(2, 2),
                                  padding='SAME', name='pool_layer1')

            _shared_net =Conv2d(_shared_net, n_filter=48, filter_size=(5, 5), strides=(1, 1),
                               act=tf.identity, padding='SAME', name='cnn2')
            _shared_net = BatchNormLayer(_shared_net,is_train=True,act=tf.nn.relu)
            _shared_net = MaxPool2d(_shared_net, filter_size=(2, 2), strides=(2, 2),
                                  padding='SAME', name='pool_layer2')
            _shared_net=FlattenLayer(_shared_net)

            feature=_shared_net.outputs

            _F1_net=_shared_net
            _F2_net=_shared_net
            _Ft_net=_shared_net

            with tf.variable_scope("F1") as scope:
                _F1_net = DropoutLayer(_F1_net, keep=drop_prob['F1'], name='drop1',is_fix=True,is_train=self.istrain)
                _F1_net=DenseLayer(_F1_net,n_units=100,
                                   act=tf.identity, name='relu1')
                _F1_net = BatchNormLayer(_F1_net,is_train=True,act=tf.nn.relu,name='bn1')
                _F1_net = DropoutLayer(_F1_net, keep=drop_prob['F1'], name='drop2',is_fix=True,is_train=self.istrain)
                _F1_net = DenseLayer(_F1_net,n_units=100,
                                   act=tf.identity, name='relu2')
                _F1_net = BatchNormLayer(_F1_net, is_train=True, act=tf.nn.relu,name='bn2')
                _F1_net = DenseLayer(_F1_net,n_units=10,
                                   act=tf.nn.softmax, name='output')
                self.F1_out=_F1_net.outputs

            with tf.variable_scope("F2") as scope:

                _F2_net = DropoutLayer(_F2_net, keep=drop_prob['F2'], name='drop1',is_fix=True,is_train=self.istrain)
                _F2_net=DenseLayer(_F2_net,n_units=100,
                                   act=tf.identity, name='relu1')
                _F2_net = BatchNormLayer(_F2_net,is_train=True,act=tf.nn.relu,name='bn1')
                _F2_net = DropoutLayer(_F2_net, keep=drop_prob['F2'], name='drop2',is_fix=True,is_train=self.istrain)
                _F2_net = DenseLayer(_F2_net,n_units=100,
                                   act=tf.identity, name='relu2')
                _F2_net = BatchNormLayer(_F2_net, is_train=True, act=tf.nn.relu,name='bn2')
                _F2_net = DenseLayer(_F2_net,n_units=10,
                                   act=tf.nn.softmax, name='output')
                self.F2_out = _F2_net.outputs

            with tf.variable_scope("Ft") as scope:
                _Ft_net = DropoutLayer(_Ft_net, keep=drop_prob['Ft'], name='drop1',is_fix=True,is_train=self.istrain)
                _Ft_net=DenseLayer(_Ft_net,n_units=100,
                                   act=tf.identity, name='relu1')
                _Ft_net = BatchNormLayer(_Ft_net,is_train=True,act=tf.nn.relu,name='bn1')
                _Ft_net = DropoutLayer(_Ft_net, keep=drop_prob['Ft'], name='drop2',is_fix=True,is_train=self.istrain)
                _Ft_net = DenseLayer(_Ft_net,n_units=100,
                                   act=tf.identity, name='relu2')
                _Ft_net = BatchNormLayer(_Ft_net, is_train=True, act=tf.nn.relu,name='bn2')
                _Ft_net = DenseLayer(_Ft_net,n_units=10,
                                   act=tf.nn.softmax, name='output')
                self.Ft_out = _Ft_net.outputs

            #self.cost = cross_entropy(F1_out,self.y_,name='F1_loss')#+cross_entropy(F2_out,self.y_,name='F2_loss')+cross_entropy(Ft_out,self.y_,name='Ft_loss')
            self.F1_loss = -tf.reduce_mean(self.y_ * tf.log(self.F1_out))
            self.F2_loss = -tf.reduce_mean(self.y_*tf.log(self.F2_out))
            self.Ft_loss = -tf.reduce_mean(self.y_*tf.log(self.Ft_out))
            self.F1_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.F1_out,1),tf.argmax(self.y_,1)),tf.float32))
            self.F2_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.F2_out, 1), tf.argmax(self.y_, 1)), tf.float32))
            self.Ft_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.Ft_out, 1), tf.argmax(self.y_, 1)), tf.float32))

            self.cost =self.F1_loss+self.F2_loss+self.Ft_loss+tf.reduce_sum(tf.abs(tf.multiply(tf.transpose(_F1_net.all_params[14]),_F2_net.all_params[14])))
            self.labeling_cost = self.F1_loss+self.F2_loss+tf.reduce_sum(tf.abs(tf.multiply(tf.transpose(_F1_net.all_params[14]),_F2_net.all_params[14])))
            self.targetspecific_cost = self.Ft_loss

            self.F1F2Ft_op = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(self.cost)
            self.F1F2_op =tf.train.AdamOptimizer(learning_rate = 0.01).minimize(self.labeling_cost)
            self.Ft_op = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(self.targetspecific_cost)


            tl.layers.initialize_global_variables(self.sess)
            print '********************************************************************************************************************************************************'
            _shared_net.print_params()
            _shared_net.print_layers()
            print '********************************************************************************************************************************************************'
            _F1_net.print_params()
            _F1_net.print_layers()
            print '********************************************************************************************************************************************************'
            _F2_net.print_params()
            _F2_net.print_layers()
            print '********************************************************************************************************************************************************'
            _Ft_net.print_params()
            _Ft_net.print_layers()




    def fit_ATDA(self,source_train, y_train, target_val, y_val, target_data, target_label, nb_epoch=5, k_epoch=100, batch_size=128, shuffle=True,N_init=5000,N_max=40000):

        n = source_train.shape[0]
        print n

        for e in range(nb_epoch):
            n_batch = 0
            for Xu_batch, Yu_batch in iterate_minibatches(source_train, y_train, batch_size, shuffle=shuffle):
                feed_dict = { self.x: Xu_batch, self.y_: Yu_batch ,self.istrain:True}
                _, cost, F1_loss, F2_loss, Ft_loss = self.sess.run([self.F1F2Ft_op, self.cost,self.F1_loss,self.F2_loss,self.Ft_loss], feed_dict=feed_dict)
                n_batch += 1
                #every 10 minibatch print loss
                if n_batch % 10==0:
                    print("Epoch %d  total_loss %f F1_loss %f F2_loss %f Ft_loss %f" % (e + 1, cost,F1_loss,F2_loss,Ft_loss))
                #every 100 minibatch eval
                if n_batch % 100==0:
                    print("**************************************************************************val_stage**************************************************************************")
                    feed_dict = {self.x: target_val, self.y_: y_val, self.istrain: False}
                    F1_acc, F2_acc, Ft_acc = self.sess.run(
                            [self.F1_acc,self.F2_acc,self.Ft_acc], feed_dict=feed_dict)
                    print("Epoch %d  F1_acc %f F2_acc %f Ft_acc %f" % (
                    e + 1,  F1_acc, F2_acc, Ft_acc))
                    print("\n*************************************************************************************************************************************************************")

        Nt,true_label=sample_candidatas(target_data, target_label, N_init)
        feed_dict = {self.x: Nt, self.istrain: False}
        F1_out, F2_out = self.sess.run(
            [self.F1_out, self.F2_out ], feed_dict=feed_dict)
        Tl,pseudo_labels,true_labels=labeling_fuc(data=Nt,output1=F1_out,output2=F2_out,true_label=true_label)
        print Tl.shape,pseudo_labels.shape,true_labels.shape
        print Tl.dtype,pseudo_labels.dtype,true_labels.dtype

        L=np.concatenate((source_train,Tl),0)
        L_label=np.concatenate((y_train,pseudo_labels),0)

        for k in range(k_epoch):
            for e in range(nb_epoch):
                n_batch = 0
                for Xu_batch, Yu_batch in iterate_minibatches(L, L_label, batch_size, shuffle=shuffle):
                    feed_dict1 = {self.x: Xu_batch, self.y_: Yu_batch, self.istrain: True}
                    _, labeling_cost = self.sess.run([self.F1F2_op, self.labeling_cost],
                                  feed_dict=feed_dict1)
                    Xu_batch,Yu_batch = sample_candidatas(Tl, pseudo_labels, batch_size)
                    feed_dict2 = {self.x: Xu_batch, self.y_: Yu_batch, self.istrain: True}
                    _ , targetspecific_cost = self.sess.run([self.Ft_op, self.targetspecific_cost],
                                  feed_dict=feed_dict2)
                    n_batch += 1
                    if n_batch % 10 == 0:
                        print("Iter %d of Epoch %d  labeling_cost %f targetspecific_cost %f" % (k + 1,
                        e + 1, labeling_cost, targetspecific_cost))

                    if n_batch % 100 == 0:
                        print(
                        "**************************************************************************val_stage**************************************************************************")
                        feed_dict = {self.x: target_val, self.y_: y_val, self.istrain: False}
                        F1_acc, F2_acc, Ft_acc = self.sess.run(
                            [self.F1_acc, self.F2_acc, self.Ft_acc], feed_dict=feed_dict)
                        print("Epoch %d  F1_acc %f F2_acc %f Ft_acc %f" % (
                            e + 1, F1_acc, F2_acc, Ft_acc))
                        print(
                        "\n*************************************************************************************************************************************************************")

            N=(k + 2) * n / 20
            print N

            if N>=N_max:
                N=N_max

            Nt, true_label = sample_candidatas(target_data, target_label, N)

            n_batch = 0
            for Xu_batch, _ in iterate_minibatches(Nt,true_label, batch_size, shuffle=False):
                feed_dict = {self.x: Xu_batch, self.istrain: False}
                F1_out_batch, F2_out_batch = self.sess.run([self.F1_out, self.F2_out], feed_dict=feed_dict)
                n_batch += 1
                if n_batch==1:
                    F1_out=F1_out_batch
                    F2_out=F2_out_batch
                else:
                    F1_out=np.vstack((F1_out,F1_out_batch))
                    F2_out=np.vstack((F2_out,F2_out_batch))


            Tl, pseudo_labels, true_labels = labeling_fuc(data=Nt, output1=F1_out, output2=F2_out,
                                                          true_label=true_label)
            print Tl.shape, pseudo_labels.shape, true_labels.shape

            L = np.concatenate((source_train, Tl), 0)
            L_label = np.concatenate((y_train, pseudo_labels), 0)















