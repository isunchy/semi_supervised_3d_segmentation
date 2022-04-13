import os
import sys
import tensorflow as tf

assert(os.path.isdir('ocnn/tensorflow'))
sys.path.append('ocnn/tensorflow')
sys.path.append('ocnn/tensorflow/script')

from libs import *
from ocnn import *


def predict_module_v3(data, num_output, num_hidden, n_layer, training=True, reuse=False, suffix=''):
  with tf.variable_scope('predict_{}{}'.format(num_output, suffix), reuse=reuse):
    for i in range(n_layer):
      with tf.variable_scope('conv{}'.format(i)):
        data = octree_conv1x1_bn_lrelu(data, num_hidden, training)
    with tf.variable_scope('conv{}'.format(n_layer)):
      logit = octree_conv1x1(data, num_output, use_bias=True)
  logit = tf.transpose(tf.squeeze(logit, [0, 3])) # (1, C, H, 1) -> (H, C)
  data = tf.transpose(tf.squeeze(data, [0, 3])) # (1, C, H, 1) -> (H, C)
  return logit, data


def extract_pts_feature_from_octree_node(inputs, octree, pts, depth, nempty=False):
  # pts shape: [n_pts, 4]
  xyz, ids = tf.split(pts, [3, 1], axis=1)
  xyz = xyz + 1.0                                             # [0, 2]
  pts_input = tf.concat([xyz * (2**(depth-1)), ids], axis=1)
  feature = octree_bilinear_v3(pts_input, inputs, octree, depth=depth, nempty=nempty)
  return feature


def network_unet(octree, depth, training=True, reuse=False, nempty=False):
  channel = 4
  nout = [512, 256, 256, 256, 256, 128, 64, 32, 16, 16, 16]
  with tf.variable_scope('ocnn_unet', reuse=reuse):    
    with tf.variable_scope('signal'):
      data = octree_property(octree, property_name='feature', dtype=tf.float32,
                             depth=depth, channel=channel)
      data = tf.abs(data)
      data = tf.reshape(data, [1, channel, -1, 1])
      if nempty:
        mask = octree_property(octree, property_name='child', dtype=tf.int32, depth=depth, channel=1) >= 0
        mask = tf.reshape(mask, [-1])
        data = tf.boolean_mask(data, mask, axis=2)

    ## encoder
    convd = [None]*10
    convd[depth+1] = data
    for d in range(depth, 1, -1):
      with tf.variable_scope('encoder_d%d' % d):
        # downsampling
        dd = d if d == depth else d + 1
        stride = 1 if d == depth else 2
        kernel_size = [3] if d == depth else [2]
        convd[d] = octree_conv_bn_relu(convd[d+1], octree, dd, nout[d], training,
                                       stride=stride, kernel_size=kernel_size, nempty=nempty)
        # resblock
        for n in range(0, 3):
          with tf.variable_scope('resblock_%d' % n):
            convd[d] = octree_resblock(convd[d], octree, d, nout[d], 1, training, nempty=nempty)

    ## decoder
    deconv = convd[2]
    for d in range(3, depth + 1):
      with tf.variable_scope('decoder_d%d' % d):
        # upsampling
        deconv = octree_deconv_bn_relu(deconv, octree, d-1, nout[d], training, 
                                       kernel_size=[2], stride=2, fast_mode=False, nempty=nempty)
        deconv = convd[d] + deconv # skip connections

        # resblock
        for n in range(0, 3):
          with tf.variable_scope('resblock_%d' % n):
            deconv = octree_resblock(deconv, octree, d, nout[d], 1, training, nempty=nempty)

  return deconv


def network_unet34(octree, depth, training=True, reuse=False, nempty=False):
  channel = 7
  encoder_nout = [32, 64, 128, 256]
  decoder_nout = [256, 128, 96, 96]
  encoder_blocks = [2, 3, 4, 6]
  decoder_blocks = [2, 2, 2, 2]

  with tf.variable_scope('ocnn_unet', reuse=reuse):    
    with tf.variable_scope('signal'):
      data = octree_property(octree, property_name='feature', dtype=tf.float32,
                            depth=depth, channel=channel)
      data = tf.reshape(data, [1, channel, -1, 1])
      data = tf.abs(data)
      if nempty:
        mask = octree_property(octree, property_name='child', dtype=tf.int32, depth=depth, channel=1) >= 0
        mask = tf.reshape(mask, [-1])
        data = tf.boolean_mask(data, mask, axis=2)

    ## encoder
    convd = [None]*10
    convd[depth] = octree_conv_bn_relu(data, octree, depth, encoder_nout[0], training, nempty=nempty)
    for i, nout in enumerate(encoder_nout):
      depth_i = depth - i - 1
      with tf.variable_scope('encoder_d%d' % depth_i):
        # downsampling
        convd[depth_i] = octree_conv_bn_relu(convd[depth_i+1], octree, depth_i+1, nout, training,
                                       stride=2, kernel_size=[2], nempty=nempty)
        # resblock
        for j in range(encoder_blocks[i]):
          with tf.variable_scope('resblock_%d' % j):
            convd[depth_i] = octree_resblock3(convd[depth_i], octree, depth_i, nout, training, nempty=nempty)
    
    depth_lowest = depth - len(encoder_nout)

    ## decoder
    deconv = convd[depth_lowest]
    for i, nout in enumerate(decoder_nout):
      depth_i = depth_lowest + i + 1
      with tf.variable_scope('decoder_d%d' % depth_i):
        # upsampling
        deconv = octree_deconv_bn_relu(deconv, octree, depth_i-1, nout, training, 
                                       kernel_size=[2], stride=2, fast_mode=False, nempty=nempty)
        deconv = tf.concat([convd[depth_i], deconv], axis=1) # skip connections
        # resblock
        for j in range(decoder_blocks[i]):
          with tf.variable_scope('resblock_%d' % j):
            deconv = octree_resblock3(deconv, octree, depth_i, nout, training, nempty=nempty)

    return deconv
