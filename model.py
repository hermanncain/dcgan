
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import os
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

def generator_simplified_api(inputs, y=None, is_train=True, reuse=False):
    image_size = 28
    gf_dim = 64 # Dimension of gen filters in first conv layer. [64]
    y_dim = 10
    c_dim = FLAGS.c_dim # n_color 1
    batch_size = FLAGS.batch_size # 64
    w_init = tf.random_normal_initializer(stddev=0.02)
    w_initl = tf.random_normal_initializer(stddev=0.02)
    b_init = tf.constant_initializer(0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    s_h, s_w = image_size, image_size
    s_h2, s_h4 = int(s_h/2), int(s_h/4)
    s_w2, s_w4 = int(s_w/2), int(s_w/4)
    gfc_dim=1024

    with tf.variable_scope("generator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        
        net_in = InputLayer(inputs, name='g/z_in')
        y = InputLayer(y,name='g/y_in')
        yb = ReshapeLayer(y, [batch_size, 1, 1, y_dim])
        z = ConcatLayer([net_in, y], 1)

        net_h0 = DenseLayer(z, n_units=gfc_dim, W_init=w_initl, b_init=b_init,
                act = tf.identity, name='g/h0/lin')
        net_h0 = BatchNormLayer(net_h0, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h0/batch_norm')
        net_h0 = ConcatLayer([net_h0, y], 1, name='g/h0/concat')

        net_h1 = DenseLayer(net_h0, n_units=gf_dim*2*s_h4*s_w4, W_init=w_initl, b_init=b_init,
                act = tf.identity, name='g/h1/lin')
        net_h1 = BatchNormLayer(net_h1, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h1/batch_norm')
        net_h1 = ReshapeLayer(net_h1, shape=[batch_size, s_h4, s_w4, gf_dim * 2], name='g/h1/reshape')
        net_h1 = condConcatLayer([net_h1, yb], 3, name='g/h1/concat') 

        net_h2 = DeConv2d(net_h1, gf_dim*2, (5, 5), out_size=(s_h2, s_w2), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h2/decon2d')
        net_h2 = BatchNormLayer(net_h2, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h2/batch_norm')
        net_h2 = condConcatLayer([net_h2, yb], 3, name='g/h2/concat') 

        net_h3 = DeConv2d(net_h2, c_dim, (5, 5), out_size=(image_size, image_size), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h3/decon2d')
        
        logits = net_h3.outputs
        net_h3.outputs = tf.nn.sigmoid(net_h3.outputs)
    return net_h3, logits

def discriminator_simplified_api(inputs, y=None, is_train=True, reuse=False):
    df_dim = 64 # Dimension of discrim filters in first conv layer. [64]
    y_dim = 10
    c_dim = FLAGS.c_dim # n_color 1
    batch_size = FLAGS.batch_size # 64
    w_init = tf.random_normal_initializer(stddev=0.02)
    w_initl = tf.random_normal_initializer(stddev=0.02)
    b_init = tf.constant_initializer(0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        
        net_in = InputLayer(inputs, name='d/in')
        y = InputLayer(y,name='g/y_in')
        yb = ReshapeLayer(y, [batch_size, 1, 1, y_dim])
        x = condConcatLayer([net_in, yb], 3, name='d/x/concat')

        net_h0 = Conv2d(x, c_dim + y_dim, (5, 5), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                padding='SAME', W_init=w_init, name='d/h0/conv2d')
        net_h0 = condConcatLayer([net_h0, yb], 3, name='d/h0/concat')

        net_h1 = Conv2d(net_h0, df_dim + y_dim, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, name='d/h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d/h1/batch_norm')
        net_h1 = ReshapeLayer(net_h1, [batch_size, -1])
        net_h1 = ConcatLayer([net_h1, y], 1, name='d/h1/concat')

        net_h2 = DenseLayer(net_h1, n_units=1024, act=tf.identity,
                W_init = w_initl, b_init=b_init, name='d/h2/lin_sigmoid')
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d/h2/batch_norm')
        net_h2 = ConcatLayer([net_h2, y], 1, name='d/h2/concat')

        net_h3 = DenseLayer(net_h2, n_units=1, act=tf.identity,
                W_init = w_initl, b_init=b_init, name='d/h3/lin_sigmoid')
        
        logits = net_h3.outputs
        net_h3.outputs = tf.nn.sigmoid(net_h3.outputs)
    return net_h3, logits

class condConcatLayer(Layer):
    def __init__(
            self,
            layers,
            concat_dim=-1,
            name='concat_layer',
    ):
        Layer.__init__(self, prev_layer=layers, name=name)
        self.inputs = []
        for l in layers:
            self.inputs.append(l.outputs)

        x = self.inputs[0]
        y = self.inputs[1]
        x_shapes = x.get_shape()
        y_shapes = y.get_shape()

        self.outputs = tf.concat([
                x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], concat_dim)

        self.all_layers.append(self.outputs)

def load_mnist():
    data_dir = os.path.join('./data', 'mnist')
    
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY)
    teY = np.asarray(teY)
    
    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)
    
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
    
    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
      y_vec[i,y[i]] = 1.0
    
    return X/255.,y_vec