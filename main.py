import os, pprint, time
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from glob import glob
from random import shuffle
from model import *
from utils import *

import win_unicode_console
win_unicode_console.enable()

pp = pprint.PrettyPrinter()

"""
TensorLayer implementation of DCGAN to generate mnist image.

Usage : see README.md
"""
flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
#flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The number of batch images [64]")
#flags.DEFINE_integer("image_size", 28, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("output_size", 28, "The size of the output images to produce [64]")
flags.DEFINE_integer("sample_size", 64, "The number of sample images [64]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_integer("sample_step", 500, "The interval of generating sample. [500]")
flags.DEFINE_integer("save_step", 500, "The interval of saveing checkpoints. [500]")
flags.DEFINE_string("dataset", "mnist", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    tl.files.exists_or_mkdir(FLAGS.checkpoint_dir)
    tl.files.exists_or_mkdir(FLAGS.sample_dir)

    z_dim = 100
    with tf.device("/gpu:0"):
        ##========================= DEFINE MODEL ===========================##
        data_X,data_y = load_mnist()
        z = tf.placeholder(tf.float32, [FLAGS.batch_size, z_dim], name='z_noise')
        y = tf.placeholder(tf.float32, [FLAGS.batch_size, 10], name='y')
        real_images =  tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim], name='real_images')

        # z --> generator for training
        net_g, g_logits = generator_simplified_api(z, y, is_train=True, reuse=False)
        # real images --> discriminator
        net_dr, dr_logits = discriminator_simplified_api(real_images, y, is_train=True, reuse=False)
        # sample_z --> generator for evaluation, set is_train to False
        # so that BatchNormLayer behave differently
        net_gs, gs_logits = generator_simplified_api(z, y, is_train=False, reuse=True)
        # generated fake images --> discriminator
        net_d, d_logits = discriminator_simplified_api(net_g.outputs, y, is_train=True, reuse=True)

        ##========================= DEFINE TRAIN OPS =======================##
        # cost for updating discriminator and generator
        # discriminator: real images are labelled as 1
        d_loss_real = tl.cost.sigmoid_cross_entropy(dr_logits, tf.ones_like(dr_logits), name='dreal')
        # discriminator: images from generator (fake) are labelled as 0
        d_loss_fake = tl.cost.sigmoid_cross_entropy(d_logits, tf.zeros_like(d_logits), name='dfake')
        d_loss = d_loss_real + d_loss_fake
        # generator: try to make the the fake images look real (1)
        g_loss = tl.cost.sigmoid_cross_entropy(d_logits, tf.ones_like(d_logits), name='gfake')

        g_vars = tl.layers.get_variables_with_name('generator', True, True)
        d_vars = tl.layers.get_variables_with_name('discriminator', True, True)

        net_g.print_params(False)
        print("---------------")
        net_d.print_params(False)

        # optimizers for updating discriminator and generator
        d_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                          .minimize(d_loss, var_list=d_vars)
        g_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                          .minimize(g_loss, var_list=g_vars)

    sess = tf.InteractiveSession()
    tl.layers.initialize_global_variables(sess)

    model_dir = "%s_%s_%s" % (FLAGS.dataset, FLAGS.batch_size, FLAGS.output_size)
    save_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)
    tl.files.exists_or_mkdir(FLAGS.sample_dir)
    tl.files.exists_or_mkdir(save_dir)
    # load the latest checkpoints
    net_g_name = os.path.join(save_dir, 'net_g.npz')
    net_d_name = os.path.join(save_dir, 'net_d.npz')

    sample_seed = np.random.uniform(-1, 1, size=(FLAGS.sample_size , z_dim))

    ##========================= TRAIN MODELS ================================##
    iter_counter = 0
    for epoch in range(FLAGS.epoch):

        sample_images = data_X[0:FLAGS.sample_size]
        sample_labels = data_y[0:FLAGS.sample_size]

        print("[*] Sample images updated!")

        ## load image data
        batch_idxs = len(data_X) // FLAGS.batch_size

        for idx in range(0, int(batch_idxs)):
            
            batch_images = data_X[idx*FLAGS.batch_size:(idx+1)*FLAGS.batch_size]
            batch_labels = data_y[idx*FLAGS.batch_size:(idx+1)*FLAGS.batch_size]
            batch_z = np.random.uniform(-1, 1, [FLAGS.batch_size, z_dim]) \
              .astype(np.float32)
            start_time = time.time()

            # updates the discriminator
            errD, _ = sess.run([d_loss, d_optim], feed_dict={z: batch_z, y:batch_labels, real_images: batch_images })
            # updates the generator, run generator twice to make sure that d_loss does not go to zero (difference from paper)
            for _ in range(2):
                errG, _ = sess.run([g_loss, g_optim], feed_dict={z: batch_z, y:batch_labels})
            print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, FLAGS.epoch, idx, batch_idxs, time.time() - start_time, errD, errG))

            iter_counter += 1
            if np.mod(iter_counter, FLAGS.sample_step) == 0:
                # generate and visualize generated images
                img, errD, errG = sess.run([net_gs.outputs, d_loss, g_loss], feed_dict={z : sample_seed, real_images: sample_images, y:sample_labels})
                tl.visualize.save_images(img, [8, 8], './{}/train_{:02d}_{:04d}.png'.format(FLAGS.sample_dir, epoch, idx))
                print("[Sample] d_loss: %.8f, g_loss: %.8f" % (errD, errG))

            if np.mod(iter_counter, FLAGS.save_step) == 0:
                # save current network parameters
                print("[*] Saving checkpoints...")
                tl.files.save_npz(net_g.all_params, name=net_g_name, sess=sess)
                tl.files.save_npz(net_d.all_params, name=net_d_name, sess=sess)
                print("[*] Saving checkpoints SUCCESS!")

if __name__ == '__main__':
    tf.app.run()
