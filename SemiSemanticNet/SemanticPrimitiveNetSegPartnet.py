import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm

sys.path.append('util')
import vis_pointcloud
from config import *
from dataset_partnet import *
from network import *
from metric import *
from tensorflow import set_random_seed
set_random_seed(2)

# batch_inbox: [bs]
def decode_batch_inbox(batch_inbox, batch_size):
  inbox_list = []
  for i in range(batch_size):
    inbox_list.append(tf.reshape(tf.decode_raw(batch_inbox[i], out_type=tf.uint8), [-1]))
  inbox = tf.concat(inbox_list, axis=0)
  return inbox

# in_mask_1: N
# in_mask_2: N
def get_intersection_mask(in_mask_1, in_mask_2):
  intersection = tf.cast(tf.multiply(in_mask_1, in_mask_2), dtype=tf.bool) # [N]
  points_1_mask = tf.boolean_mask(intersection, in_mask_1>0) # [N1]
  points_2_mask = tf.boolean_mask(intersection, in_mask_2>0) # [N2]
  return points_1_mask, points_2_mask

def get_split_label(label):
  label_1 = tf.cast(tf.math.floordiv(label, 10000), dtype=tf.int32)
  label_2 = tf.cast(tf.math.floormod(tf.math.floordiv(label, 100), 100), dtype=tf.int32)
  label_3 = tf.cast(tf.math.floormod(label, 100), dtype=tf.int32)
  return label_1, label_2, label_3

def get_training_input_data(dataset_path, batch_size, depth=6):
  # N_point: point number of self after clip
  # N_ghost_point: point number of ghost after clip
  # N_POINT: total point number of batch shape before clip
  with tf.name_scope('input_data_training'):
    [octree, ghost_octree, inbox_points_bytes, ghost_inbox_points_bytes, inbox_mask_bytes,
        ghost_inbox_mask_bytes, points_flag] = points_dataset(dataset_path, batch_size, depth=depth,
            test=False, rotation_angle=FLAGS.rotation_angle, scale=FLAGS.scale, jitter=FLAGS.jitter)

    inbox_node_position = points_property(inbox_points_bytes, property_name='xyz', channel=4) # [N_point, 4]
    inbox_batch_index = tf.cast(inbox_node_position[:, -1], dtype=tf.int32) # [N_point]

    ghost_inbox_node_position = points_property(ghost_inbox_points_bytes, property_name='xyz', channel=4) # [N_ghost_point, 4]
    ghost_inbox_batch_index = tf.cast(ghost_inbox_node_position[:, -1], dtype=tf.int32) # [N_ghost_point]

    inbox_gt_part_index = points_property(inbox_points_bytes, property_name='label', channel=1) # [N_point, 1]
    inbox_gt_part_index = tf.reshape(tf.cast(inbox_gt_part_index, dtype=tf.int32), [-1]) # [N_point]
    inbox_gt_part_index_1, inbox_gt_part_index_2, inbox_gt_part_index_3 = get_split_label(inbox_gt_part_index) # [N_point, N_point, N_point]

    ghost_inbox_gt_part_index = points_property(ghost_inbox_points_bytes, property_name='label', channel=1) # [N_ghost_point, 1]
    ghost_inbox_gt_part_index = tf.reshape(tf.cast(ghost_inbox_gt_part_index, dtype=tf.int32), [-1]) # [N_ghost_point]
    ghost_inbox_gt_part_index_1, ghost_inbox_gt_part_index_2, ghost_inbox_gt_part_index_3 = get_split_label(ghost_inbox_gt_part_index) # [N_ghost_point, N_ghost_point, N_ghost_point]

    inbox_mask = decode_batch_inbox(inbox_mask_bytes, batch_size) # [N_POINT]
    ghost_inbox_mask = decode_batch_inbox(ghost_inbox_mask_bytes, batch_size) # [N_POINT]

    inter_mask, ghost_inter_mask = get_intersection_mask(inbox_mask, ghost_inbox_mask) # [N_point], [N_ghost_point]
    points_flag = tf.cast(tf.reshape(points_flag, [-1]), tf.float32) # [bs], labeled data flag

  return [octree, ghost_octree, inbox_node_position, ghost_inbox_node_position, inbox_batch_index, ghost_inbox_batch_index,
      inbox_gt_part_index_1, inbox_gt_part_index_2, inbox_gt_part_index_3, ghost_inbox_gt_part_index_1,
      ghost_inbox_gt_part_index_2, ghost_inbox_gt_part_index_3, inter_mask, ghost_inter_mask, points_flag]

def get_test_input_data(dataset_path, batch_size, depth=6):
  with tf.name_scope('input_data_training'):
    [octree, points_bytes] = points_dataset(dataset_path, batch_size, depth=depth, test=True)
    node_position = points_property(points_bytes, property_name='xyz', channel=4) # [n_point, 4]
    point_gt_part_index = points_property(points_bytes, property_name='label', channel=1) # [n_point, 1]
    point_gt_part_index = tf.reshape(tf.cast(point_gt_part_index, dtype=tf.int32), [-1]) # [n_point]
    point_gt_part_index_1, point_gt_part_index_2, point_gt_part_index_3 = get_split_label(point_gt_part_index) # [n_point, n_point, n_point]
    point_batch_index = tf.cast(node_position[:, -1], dtype=tf.int32) # [n_point]
  return octree, node_position, point_gt_part_index_1, point_gt_part_index_2, point_gt_part_index_3, point_batch_index

# node_position: [n_point, 4]
def backbone(octree, node_position, depth=6, training=True, reuse=False, nempty=False):
  node_feature = network_unet(octree, depth, training=training, reuse=reuse, nempty=nempty) # [1, C, n_node, 1]
  point_feature = extract_pts_feature_from_octree_node(node_feature, octree, node_position, depth, nempty=nempty) # [1, C, n_point, 1]
  return point_feature

# point_feature: [1, C, n_point, 1]
def seg_header(point_feature, n_part, n_hidden=128, n_layer=2, training=True, reuse=False, suffix=''):
  point_predict_logits, point_hidden_feature = predict_module_v3(
      point_feature, n_part, n_hidden, n_layer, training=training, reuse=reuse, suffix=suffix) # [n_point, n_part], [n_point, n_hidden]
  point_predict_prob = tf.nn.softmax(point_predict_logits) # [n_point, n_part]
  return point_predict_logits, point_predict_prob, point_hidden_feature

# point_predict_logits: [n_point, n_part]
# point_gt_part_index: [n_point]
# inbox_batch_index: [n_point]
def compute_segmentation_loss(point_predict_logits, point_gt_part_index, inbox_batch_index, n_part, points_flag=None, delete_0=False):
  with tf.name_scope('segmentation_loss'):
    seg_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(point_gt_part_index, n_part, dtype=tf.float32),
        logits=point_predict_logits) # [n_point]
    if delete_0:
      non_zero_mask = tf.cast(point_gt_part_index > 0, dtype=tf.float32) # [n_point]
      seg_loss = tf.math.divide_no_nan(
          tf.segment_sum(tf.multiply(seg_loss, non_zero_mask), inbox_batch_index),
          tf.segment_sum(non_zero_mask, inbox_batch_index)) # [bs]
    else:
      non_zero_mask = tf.cast(point_gt_part_index >= 0, dtype=tf.float32) # [n_point]
      seg_loss = tf.math.divide_no_nan(
          tf.segment_sum(tf.multiply(seg_loss, non_zero_mask), inbox_batch_index),
          tf.segment_sum(non_zero_mask, inbox_batch_index)) # [bs]
  if points_flag is None:
    return tf.reduce_mean(seg_loss)
  else:
    su_seg_loss = tf.math.divide_no_nan(
        tf.reduce_sum(tf.multiply(seg_loss, points_flag)), tf.reduce_sum(points_flag)) # scalar
    un_seg_loss = tf.math.divide_no_nan(
        tf.reduce_sum(tf.multiply(seg_loss, 1-points_flag)), tf.reduce_sum(1-points_flag)) # scalar
    return su_seg_loss, un_seg_loss

# point_predict_part_index: [n_point]
# point_gt_part_index: [n_point]
# point_batch_index: [n_point]
def compute_point_match_accuracy(point_predict_part_index, point_gt_part_index, point_batch_index, delete_0=False):
  with tf.name_scope('point_match_accuracy'):
    point_predict_match = tf.cast(tf.math.equal(point_predict_part_index, point_gt_part_index), dtype=tf.float32) # [n_point]
    if delete_0:
      non_zero_mask = tf.cast(point_gt_part_index > 0, dtype=tf.float32) # [n_point]
      point_match_accuracy = tf.math.divide_no_nan(
          tf.segment_sum(tf.multiply(point_predict_match, non_zero_mask), point_batch_index),
          tf.segment_sum(non_zero_mask, point_batch_index)) # [bs]
    else:
      non_zero_mask = tf.cast(point_gt_part_index >= 0, dtype=tf.float32) # [n_point]
      point_match_accuracy = tf.math.divide_no_nan(
          tf.segment_sum(tf.multiply(point_predict_match, non_zero_mask), point_batch_index),
          tf.segment_sum(non_zero_mask, point_batch_index)) # [bs]
  return point_match_accuracy

# point_predict_prob: [n_point, n_part]
# point_gt_part_index: [n_point]
# point_batch_index: [n_point]
def compute_segmentation_accuracy(point_predict_prob, point_gt_part_index, point_batch_index, points_flag=None, delete_0=False):
  with tf.name_scope('segmentation_accuracy'):
    point_predict_part_index = tf.argmax(point_predict_prob, axis=-1, output_type=tf.int32) # [n_point]
    point_match_accuracy = compute_point_match_accuracy(point_predict_part_index,
        point_gt_part_index, point_batch_index, delete_0=delete_0) # [bs]
  if points_flag is None:
    return tf.reduce_mean(point_match_accuracy), point_predict_part_index
  else:
    su_seg_accuracy = tf.math.divide_no_nan(
        tf.reduce_sum(tf.multiply(point_match_accuracy, points_flag)), tf.reduce_sum(points_flag)) # scalar
    un_seg_accuracy = tf.math.divide_no_nan(
        tf.reduce_sum(tf.multiply(point_match_accuracy, 1-points_flag)), tf.reduce_sum(1-points_flag)) # scalar
    return su_seg_accuracy, un_seg_accuracy, point_predict_part_index

# point_predict_prob: [n_point, n_part]
# ghost_point_predict_prob: [n_point, n_part]
# point_batch_index: [n_point]
def compute_point_consistency_loss(point_predict_prob, ghost_point_predict_prob, point_batch_index):
  with tf.name_scope('point_consistency_loss'):
    if FLAGS.use_kl:
      # kl loss
      point_consistency_loss = 0.5*(
          tf.reduce_sum(tf.math.xlogy(point_predict_prob,
              tf.math.divide_no_nan(point_predict_prob, tf.clip_by_value(ghost_point_predict_prob, 1e-7, 1.-1e-7))+1e-7), axis=-1) +
          tf.reduce_sum(tf.math.xlogy(ghost_point_predict_prob,
              tf.math.divide_no_nan(ghost_point_predict_prob, tf.clip_by_value(point_predict_prob, 1e-7, 1.-1e-7))+1e-7), axis=-1)
      ) # [n_point]
    else:
      # mse loss
      point_consistency_diff = point_predict_prob - ghost_point_predict_prob # [n_point, n_part]
      point_consistency_loss = tf.reduce_sum(point_consistency_diff**2, axis=1) # [n_point]

    point_consistency_loss = tf.segment_mean(point_consistency_loss, point_batch_index) # [bs]
    point_consistency_loss = tf.reduce_mean(point_consistency_loss) # scalar
  return point_consistency_loss

# point_predict_part_index: [n_point]
# ghost_point_predict_part_index: [n_point]
# point_batch_index: [n_point]
def compute_point_consistency_accuracy(point_predict_part_index, ghost_point_predict_part_index, point_batch_index):
  with tf.name_scope('point_consistency_accuracy'):
    point_match_accuracy = compute_point_match_accuracy(point_predict_part_index, ghost_point_predict_part_index, point_batch_index) # [bs]
    point_consistency_accuracy = tf.reduce_mean(point_match_accuracy) # scalar
  return point_consistency_accuracy

# point_predict_part_index: [n_point]
# point_predict_prob: [n_point, n_part]
# ghost_point_predict_prob: [n_point, n_part]
# point_batch_index: [n_point]
def compute_part_consistency_loss(point_predict_part_index, point_predict_prob, ghost_point_predict_prob, point_batch_index, n_part, delete_0=False):
  with tf.name_scope('part_consistency_loss'):
    point_predict_part_index_of_part = tf.one_hot(point_predict_part_index, n_part, dtype=tf.float32) # [n_point, n_part]
    point_predict_part_index_of_nopart = 1 - point_predict_part_index_of_part # [n_point, n_part]
    if delete_0:
      point_predict_part_index_of_part = point_predict_part_index_of_part[:, 1:] # [n_point, n_part-1]
      point_predict_part_index_of_nopart = point_predict_part_index_of_nopart[:, 1:] # [n_point, n_part-1]
      point_predict_prob = point_predict_prob[:, 1:] # [n_point, n_part-1]
      ghost_point_predict_prob = ghost_point_predict_prob[:, 1:] # [n_point, n_part-1]
    point_predict_prob_of_part = tf.multiply(point_predict_prob, point_predict_part_index_of_part) # [n_point, n_part]
    point_predict_prob_of_nopart = tf.multiply(point_predict_prob, point_predict_part_index_of_nopart) # [n_point, n_part]
    ghost_point_predict_prob_of_part = tf.multiply(ghost_point_predict_prob, point_predict_part_index_of_part) # [n_point, n_part]
    ghost_point_predict_prob_of_nopart = tf.multiply(ghost_point_predict_prob, point_predict_part_index_of_nopart) # [n_point, n_part]
    point_predict_number_of_part = tf.segment_sum(point_predict_part_index_of_part, point_batch_index) # [bs, n_part]
    point_predict_number_of_nopart = tf.segment_sum(point_predict_part_index_of_nopart, point_batch_index) # [bs, n_part]
    point_predict_prob_of_part_mean = tf.math.divide_no_nan(
        tf.segment_sum(point_predict_prob_of_part, point_batch_index), point_predict_number_of_part) # [bs, n_part]
    point_predict_prob_of_nopart_mean = tf.math.divide_no_nan(
        tf.segment_sum(point_predict_prob_of_nopart, point_batch_index), point_predict_number_of_nopart) # [bs, n_part]
    ghost_point_predict_prob_of_part_mean = tf.math.divide_no_nan(
        tf.segment_sum(ghost_point_predict_prob_of_part, point_batch_index), point_predict_number_of_part) # [bs, n_part]
    ghost_point_predict_prob_of_nopart_mean = tf.math.divide_no_nan(
        tf.segment_sum(ghost_point_predict_prob_of_nopart, point_batch_index), point_predict_number_of_nopart) # [bs, n_part]
    part_consistency_loss = (point_predict_prob_of_part_mean - ghost_point_predict_prob_of_part_mean)**2 + \
        (point_predict_prob_of_nopart_mean - ghost_point_predict_prob_of_nopart_mean)**2 # [bs, n_part]
    point_predict_part_component = tf.cast(point_predict_number_of_part>=1, dtype=tf.float32) + tf.cast(point_predict_number_of_nopart>=1, dtype=tf.float32) # [bs, n_part]
    part_consistency_loss = tf.reduce_mean(tf.math.divide_no_nan(part_consistency_loss, point_predict_part_component), axis=1) # [bs]
    part_consistency_loss = tf.reduce_mean(part_consistency_loss) # scalar
  return part_consistency_loss

# part_predict_mask: [bs, n_part]
# part_gt_mask: [bs, n_part]
def compute_part_match_accuracy(part_predict_mask, part_gt_mask, delete_0=False):
  with tf.name_scope('part_match_accuracy'):
    part_predict_match = tf.cast(tf.math.equal(part_predict_mask, part_gt_mask), dtype=tf.float32) # [bs, n_part]
    if delete_0:
      part_predict_match = part_predict_match[:, 1:] # [bs, n_part-1]
    part_match_accuracy = tf.reduce_mean(part_predict_match, axis=1) # [bs]
  return part_match_accuracy

# point_predict_part_index: [n_point]
# ghost_point_predict_part_index: [n_point]
# point_batch_index: [n_point]
def compute_part_consistency_accuracy(point_predict_part_index, ghost_point_predict_part_index, point_batch_index, n_part, delete_0=False):
  with tf.name_scope('part_consistency_accuracy'):
    part_predict_mask = tf.cast(tf.segment_sum(tf.one_hot(point_predict_part_index, n_part, dtype=tf.int32), point_batch_index) >= 1, dtype=tf.int32) # [bs, n_part]
    ghost_part_predict_mask = tf.cast(tf.segment_sum(tf.one_hot(ghost_point_predict_part_index, n_part, dtype=tf.int32), point_batch_index) >= 1, dtype=tf.int32) # [bs, n_part]
    part_match_accuracy = compute_part_match_accuracy(part_predict_mask, ghost_part_predict_mask, delete_0=delete_0) # [bs]
    part_consistency_accuracy = tf.reduce_mean(part_match_accuracy) # scalar
  return part_consistency_accuracy

# point_predict_part_index: [n_point]
# point_gt_part_index: [n_point]
# point_batch_index: [n_point]
def compute_part_predict_accuracy(point_predict_part_index, point_gt_part_index, point_batch_index, n_part, delete_0=False):
  with tf.name_scope('part_predict_accuracy'):
    part_predict_mask = tf.cast(tf.segment_sum(tf.one_hot(point_predict_part_index, n_part, dtype=tf.int32), point_batch_index) >= 1, dtype=tf.int32) # [bs, n_part]
    part_gt_mask = tf.cast(tf.segment_sum(tf.one_hot(point_gt_part_index, n_part, dtype=tf.int32), point_batch_index) >= 1, dtype=tf.int32) # [bs, n_part]
    part_match_accuracy = compute_part_match_accuracy(part_predict_mask, part_gt_mask, delete_0=delete_0) # [bs]
    part_predict_accuracy = tf.reduce_mean(part_match_accuracy) # scalar
  return part_predict_accuracy, part_predict_mask, part_gt_mask

def train_network():
  [
      octree,                           # string
      ghost_octree,                     # string
      inbox_node_position,              # [N_point, 4]
      ghost_inbox_node_position,        # [N_ghost_point, 4]
      inbox_batch_index,                # [N_point]
      ghost_inbox_batch_index,          # [N_ghost_point]
      inbox_gt_part_index_1,            # [N_point]
      inbox_gt_part_index_2,            # [N_point]
      inbox_gt_part_index_3,            # [N_point]
      ghost_inbox_gt_part_index_1,      # [N_ghost_point]
      ghost_inbox_gt_part_index_2,      # [N_ghost_point]
      ghost_inbox_gt_part_index_3,      # [N_ghost_point]
      inter_mask,                       # [N_point]
      ghost_inter_mask,                 # [N_ghost_point]
      points_flag                       # [bs]
  ] = get_training_input_data(FLAGS.train_data, FLAGS.train_batch_size, depth=6)

  # get point feature
  with tf.variable_scope('seg'):
    with tf.name_scope('self'):
      inbox_point_feature = backbone(octree, inbox_node_position, training=True, reuse=False, nempty=FLAGS.nempty) # [1, C, N_point, 1]

      inbox_point_predict_logits_1, inbox_point_predict_prob_1, _ = seg_header(
          inbox_point_feature, n_part_1, training=True, reuse=False, suffix='_level_1') # [N_point, n_part_1], [N_point, n_part_1]
      inter_point_predict_logits_1 = tf.boolean_mask(inbox_point_predict_logits_1, inter_mask) # [N_inter, n_part_1]
      inter_point_predict_prob_1 = tf.boolean_mask(inbox_point_predict_prob_1, inter_mask) # [N_inter, n_part_1]

      inbox_point_predict_logits_2, inbox_point_predict_prob_2, _ = seg_header(
          inbox_point_feature, n_part_2, training=True, reuse=False, suffix='_level_2') # [N_point, n_part_2], [N_point, n_part_2]
      inter_point_predict_logits_2 = tf.boolean_mask(inbox_point_predict_logits_2, inter_mask) # [N_inter, n_part_2]
      inter_point_predict_prob_2 = tf.boolean_mask(inbox_point_predict_prob_2, inter_mask) # [N_inter, n_part_2]

      inbox_point_predict_logits_3, inbox_point_predict_prob_3, _ = seg_header(
          inbox_point_feature, n_part_3, training=True, reuse=False, suffix='_level_3') # [N_point, n_part_3], [N_point, n_part_3]
      inter_point_predict_logits_3 = tf.boolean_mask(inbox_point_predict_logits_3, inter_mask) # [N_inter, n_part_3]
      inter_point_predict_prob_3 = tf.boolean_mask(inbox_point_predict_prob_3, inter_mask) # [N_inter, n_part_3]

      inter_batch_index = tf.boolean_mask(inbox_batch_index, inter_mask) # [N_inter]

    with tf.name_scope('ghost'):
      ghost_inbox_point_feature = backbone(ghost_octree, ghost_inbox_node_position, training=True, reuse=True, nempty=FLAGS.nempty) # [1, C, N_ghost_point, 1]

      ghost_inbox_point_predict_logits_1, ghost_inbox_point_predict_prob_1, _ = seg_header(
          ghost_inbox_point_feature, n_part_1, training=True, reuse=True, suffix='_level_1') # [N_ghost_point, n_part_1], [N_ghost_point, n_part_1]
      ghost_inter_point_predict_logits_1 = tf.boolean_mask(ghost_inbox_point_predict_logits_1, ghost_inter_mask) # [N_inter, n_part_1]
      ghost_inter_point_predict_prob_1 = tf.boolean_mask(ghost_inbox_point_predict_prob_1, ghost_inter_mask) # [N_inter, n_part_1]

      ghost_inbox_point_predict_logits_2, ghost_inbox_point_predict_prob_2, _ = seg_header(
          ghost_inbox_point_feature, n_part_2, training=True, reuse=True, suffix='_level_2') # [N_ghost_point, n_part_2], [N_ghost_point, n_part_2]
      ghost_inter_point_predict_logits_2 = tf.boolean_mask(ghost_inbox_point_predict_logits_2, ghost_inter_mask) # [N_inter, n_part_2]
      ghost_inter_point_predict_prob_2 = tf.boolean_mask(ghost_inbox_point_predict_prob_2, ghost_inter_mask) # [N_inter, n_part_2]

      ghost_inbox_point_predict_logits_3, ghost_inbox_point_predict_prob_3, _ = seg_header(
          ghost_inbox_point_feature, n_part_3, training=True, reuse=True, suffix='_level_3') # [N_ghost_point, n_part_3], [N_ghost_point, n_part_3]
      ghost_inter_point_predict_logits_3 = tf.boolean_mask(ghost_inbox_point_predict_logits_3, ghost_inter_mask) # [N_inter, n_part_3]
      ghost_inter_point_predict_prob_3 = tf.boolean_mask(ghost_inbox_point_predict_prob_3, ghost_inter_mask) # [N_inter, n_part_3]

      ghost_inter_batch_index = tf.boolean_mask(ghost_inbox_batch_index, ghost_inter_mask) # [N_inter]

    with tf.control_dependencies([tf.assert_equal(inter_batch_index, ghost_inter_batch_index)]):
      inter_batch_index = inter_batch_index

  # segmentation loss
  with tf.name_scope('seg_loss'):
    with tf.name_scope('loss'):
      with tf.name_scope('self'):
        self_su_seg_loss_1, self_un_seg_loss_1 = compute_segmentation_loss(
            inbox_point_predict_logits_1, inbox_gt_part_index_1, inbox_batch_index, n_part_1, points_flag=points_flag, delete_0=FLAGS.delete_0) # scalar, scalar
        self_su_seg_loss_2, self_un_seg_loss_2 = compute_segmentation_loss(
            inbox_point_predict_logits_2, inbox_gt_part_index_2, inbox_batch_index, n_part_2, points_flag=points_flag, delete_0=FLAGS.delete_0) # scalar, scalar
        self_su_seg_loss_3, self_un_seg_loss_3 = compute_segmentation_loss(
            inbox_point_predict_logits_3, inbox_gt_part_index_3, inbox_batch_index, n_part_3, points_flag=points_flag, delete_0=FLAGS.delete_0) # scalar, scalar
      with tf.name_scope('ghost'):
        ghost_su_seg_loss_1, ghost_un_seg_loss_1 = compute_segmentation_loss(
            ghost_inbox_point_predict_logits_1, ghost_inbox_gt_part_index_1, ghost_inbox_batch_index, n_part_1, points_flag=points_flag, delete_0=FLAGS.delete_0) # scalar, scalar
        ghost_su_seg_loss_2, ghost_un_seg_loss_2 = compute_segmentation_loss(
            ghost_inbox_point_predict_logits_2, ghost_inbox_gt_part_index_2, ghost_inbox_batch_index, n_part_2, points_flag=points_flag, delete_0=FLAGS.delete_0) # scalar, scalar
        ghost_su_seg_loss_3, ghost_un_seg_loss_3 = compute_segmentation_loss(
            ghost_inbox_point_predict_logits_3, ghost_inbox_gt_part_index_3, ghost_inbox_batch_index, n_part_3, points_flag=points_flag, delete_0=FLAGS.delete_0) # scalar, scalar
      with tf.name_scope('total'):
        su_seg_loss_1 = 0.5*(self_su_seg_loss_1 + ghost_su_seg_loss_1) # scalar
        un_seg_loss_1 = 0.5*(self_un_seg_loss_1 + ghost_un_seg_loss_1) # scalar
        su_seg_loss_2 = 0.5*(self_su_seg_loss_2 + ghost_su_seg_loss_2) # scalar
        un_seg_loss_2 = 0.5*(self_un_seg_loss_2 + ghost_un_seg_loss_2) # scalar
        su_seg_loss_3 = 0.5*(self_su_seg_loss_3 + ghost_su_seg_loss_3) # scalar
        un_seg_loss_3 = 0.5*(self_un_seg_loss_3 + ghost_un_seg_loss_3) # scalar
    with tf.name_scope('accuracy'):
      with tf.name_scope('self'):
        su_seg_accuracy_1, un_seg_accuracy_1, inbox_point_predict_part_index_1 = compute_segmentation_accuracy(
            inbox_point_predict_prob_1, inbox_gt_part_index_1, inbox_batch_index, points_flag=points_flag, delete_0=FLAGS.delete_0) # scalar, scalar, [N_point]
        inter_point_predict_part_index_1 = tf.boolean_mask(inbox_point_predict_part_index_1, inter_mask) # [N_inter]
        su_seg_accuracy_2, un_seg_accuracy_2, inbox_point_predict_part_index_2 = compute_segmentation_accuracy(
            inbox_point_predict_prob_2, inbox_gt_part_index_2, inbox_batch_index, points_flag=points_flag, delete_0=FLAGS.delete_0) # scalar, scalar, [N_point]
        inter_point_predict_part_index_2 = tf.boolean_mask(inbox_point_predict_part_index_2, inter_mask) # [N_inter]
        su_seg_accuracy_3, un_seg_accuracy_3, inbox_point_predict_part_index_3 = compute_segmentation_accuracy(
            inbox_point_predict_prob_3, inbox_gt_part_index_3, inbox_batch_index, points_flag=points_flag, delete_0=FLAGS.delete_0) # scalar, scalar, [N_point]
        inter_point_predict_part_index_3 = tf.boolean_mask(inbox_point_predict_part_index_3, inter_mask) # [N_inter]
      with tf.name_scope('ghost'):
        ghost_su_seg_accuracy_1, ghost_un_seg_accuracy_1, ghost_inbox_point_predict_part_index_1 = compute_segmentation_accuracy(
            ghost_inbox_point_predict_prob_1, ghost_inbox_gt_part_index_1, ghost_inbox_batch_index, points_flag=points_flag, delete_0=FLAGS.delete_0) # scalar, scalar, [N_ghost_point]
        ghost_inter_point_predict_part_index_1 = tf.boolean_mask(ghost_inbox_point_predict_part_index_1, ghost_inter_mask) # [N_inter]
        ghost_su_seg_accuracy_2, ghost_un_seg_accuracy_2, ghost_inbox_point_predict_part_index_2 = compute_segmentation_accuracy(
            ghost_inbox_point_predict_prob_2, ghost_inbox_gt_part_index_2, ghost_inbox_batch_index, points_flag=points_flag, delete_0=FLAGS.delete_0) # scalar, scalar, [N_ghost_point]
        ghost_inter_point_predict_part_index_2 = tf.boolean_mask(ghost_inbox_point_predict_part_index_2, ghost_inter_mask) # [N_inter]
        ghost_su_seg_accuracy_3, ghost_un_seg_accuracy_3, ghost_inbox_point_predict_part_index_3 = compute_segmentation_accuracy(
            ghost_inbox_point_predict_prob_3, ghost_inbox_gt_part_index_3, ghost_inbox_batch_index, points_flag=points_flag, delete_0=FLAGS.delete_0) # scalar, scalar, [N_ghost_point]
        ghost_inter_point_predict_part_index_3 = tf.boolean_mask(ghost_inbox_point_predict_part_index_3, ghost_inter_mask) # [N_inter]

  # point consistency loss
  with tf.name_scope('point_consistency_loss'):
    with tf.name_scope('loss'):
      point_consistency_loss_1 = compute_point_consistency_loss(inter_point_predict_prob_1, ghost_inter_point_predict_prob_1, inter_batch_index) # scalar
      point_consistency_loss_2 = compute_point_consistency_loss(inter_point_predict_prob_2, ghost_inter_point_predict_prob_2, inter_batch_index) # scalar
      point_consistency_loss_3 = compute_point_consistency_loss(inter_point_predict_prob_3, ghost_inter_point_predict_prob_3, inter_batch_index) # scalar

    with tf.name_scope('accuracy'):
      point_consistency_accuracy_1 = compute_point_consistency_accuracy(inter_point_predict_part_index_1, ghost_inter_point_predict_part_index_1, inter_batch_index) # scalar
      point_consistency_accuracy_2 = compute_point_consistency_accuracy(inter_point_predict_part_index_2, ghost_inter_point_predict_part_index_2, inter_batch_index) # scalar
      point_consistency_accuracy_3 = compute_point_consistency_accuracy(inter_point_predict_part_index_3, ghost_inter_point_predict_part_index_3, inter_batch_index) # scalar

  # part consistency loss
  with tf.name_scope('part_consistency_loss'):
    with tf.name_scope('loss'):
      with tf.name_scope('self'):
        self_part_consistency_loss_1 = compute_part_consistency_loss(inter_point_predict_part_index_1,
            inter_point_predict_prob_1, ghost_inter_point_predict_prob_1, inter_batch_index, n_part_1, delete_0=FLAGS.delete_0) # scalar
        self_part_consistency_loss_2 = compute_part_consistency_loss(inter_point_predict_part_index_2,
            inter_point_predict_prob_2, ghost_inter_point_predict_prob_2, inter_batch_index, n_part_2, delete_0=FLAGS.delete_0) # scalar
        self_part_consistency_loss_3 = compute_part_consistency_loss(inter_point_predict_part_index_3,
            inter_point_predict_prob_3, ghost_inter_point_predict_prob_3, inter_batch_index, n_part_3, delete_0=FLAGS.delete_0) # scalar
      with tf.name_scope('ghost'):
        ghost_part_consistency_loss_1 = compute_part_consistency_loss(ghost_inter_point_predict_part_index_1,
            ghost_inter_point_predict_prob_1, inter_point_predict_prob_1, ghost_inter_batch_index, n_part_1, delete_0=FLAGS.delete_0) # scalar
        ghost_part_consistency_loss_2 = compute_part_consistency_loss(ghost_inter_point_predict_part_index_2,
            ghost_inter_point_predict_prob_2, inter_point_predict_prob_2, ghost_inter_batch_index, n_part_2, delete_0=FLAGS.delete_0) # scalar
        ghost_part_consistency_loss_3 = compute_part_consistency_loss(ghost_inter_point_predict_part_index_3,
            ghost_inter_point_predict_prob_3, inter_point_predict_prob_3, ghost_inter_batch_index, n_part_3, delete_0=FLAGS.delete_0) # scalar
      with tf.name_scope('total'):
        part_consistency_loss_1 = 0.5*(self_part_consistency_loss_1 + ghost_part_consistency_loss_1) # scalar
        part_consistency_loss_2 = 0.5*(self_part_consistency_loss_2 + ghost_part_consistency_loss_2) # scalar
        part_consistency_loss_3 = 0.5*(self_part_consistency_loss_3 + ghost_part_consistency_loss_3) # scalar
    with tf.name_scope('accuracy'):
      part_consistency_accuracy_1 = compute_part_consistency_accuracy(
          inter_point_predict_part_index_1, ghost_inter_point_predict_part_index_1, inter_batch_index, n_part_1, delete_0=FLAGS.delete_0) # scalar
      part_consistency_accuracy_2 = compute_part_consistency_accuracy(
          inter_point_predict_part_index_2, ghost_inter_point_predict_part_index_2, inter_batch_index, n_part_2, delete_0=FLAGS.delete_0) # scalar
      part_consistency_accuracy_3 = compute_part_consistency_accuracy(
          inter_point_predict_part_index_3, ghost_inter_point_predict_part_index_3, inter_batch_index, n_part_3, delete_0=FLAGS.delete_0) # scalar

  # hierarchy point consistency loss
  with tf.name_scope('hierarchy_point_consistency_loss'):
    with tf.name_scope('mapping'):
      with tf.name_scope('3-2'):
        label_mapping_32 = get_label_mapping(category_label[FLAGS.category], 0) # [n_part_3]
        label_mapping_32_matrix = tf.one_hot(label_mapping_32, n_part_2, dtype=tf.float32) # [n_part_3, n_part_2]
      with tf.name_scope('2-1'):
        label_mapping_21 = get_label_mapping(category_label[FLAGS.category], 1) # [n_part_2]
        label_mapping_21_matrix = tf.one_hot(label_mapping_21, n_part_1, dtype=tf.float32) # [n_part_2, n_part_1]
    with tf.name_scope('prob_merge'):
      with tf.name_scope('self'):
        inter_point_predict_prob_32 = tf.matmul(inter_point_predict_prob_3, label_mapping_32_matrix) # [N_inter, n_part_2]
        inter_point_predict_prob_21 = tf.matmul(inter_point_predict_prob_2, label_mapping_21_matrix) # [N_inter, n_part_1]
      with tf.name_scope('ghost'):
        ghost_inter_point_predict_prob_32 = tf.matmul(ghost_inter_point_predict_prob_3, label_mapping_32_matrix) # [N_inter, n_part_2]
        ghost_inter_point_predict_prob_21 = tf.matmul(ghost_inter_point_predict_prob_2, label_mapping_21_matrix) # [N_inter, n_part_1]
    with tf.name_scope('loss'):
      with tf.name_scope('self'):
        self_hierarchy_point_consistency_loss_32 = compute_point_consistency_loss(
            inter_point_predict_prob_32, ghost_inter_point_predict_prob_2, inter_batch_index) # scalar
        self_hierarchy_point_consistency_loss_21 = compute_point_consistency_loss(
            inter_point_predict_prob_21, ghost_inter_point_predict_prob_1, inter_batch_index) # scalar
      with tf.name_scope('ghost'):
        ghost_hierarchy_point_consistency_loss_32 = compute_point_consistency_loss(
            ghost_inter_point_predict_prob_32, inter_point_predict_prob_2, ghost_inter_batch_index) # scalar
        ghost_hierarchy_point_consistency_loss_21 = compute_point_consistency_loss(
            ghost_inter_point_predict_prob_21, inter_point_predict_prob_1, ghost_inter_batch_index) # scalar
      with tf.name_scope('total'):
        hierarchy_point_consistency_loss_32 = 0.5*(self_hierarchy_point_consistency_loss_32 + ghost_hierarchy_point_consistency_loss_32) # scalar
        hierarchy_point_consistency_loss_21 = 0.5*(self_hierarchy_point_consistency_loss_21 + ghost_hierarchy_point_consistency_loss_21) # scalar

  # total loss
  level_1_loss = FLAGS.seg_loss_weight * su_seg_loss_1 + \
      FLAGS.point_consistency_weight * point_consistency_loss_1 + \
      FLAGS.part_consistency_weight * part_consistency_loss_1

  level_2_loss = FLAGS.seg_loss_weight * su_seg_loss_2 + \
      FLAGS.point_consistency_weight * point_consistency_loss_2 + \
      FLAGS.part_consistency_weight * part_consistency_loss_2

  level_3_loss = FLAGS.seg_loss_weight * su_seg_loss_3 + \
      FLAGS.point_consistency_weight * point_consistency_loss_3 + \
      FLAGS.part_consistency_weight * part_consistency_loss_3

  level_32_loss = FLAGS.hierarchy_point_consistency_weight * hierarchy_point_consistency_loss_32

  level_21_loss = FLAGS.hierarchy_point_consistency_weight * hierarchy_point_consistency_loss_21

  train_loss = \
      level_1_loss * FLAGS.level_1_weight + \
      level_2_loss * FLAGS.level_2_weight + \
      level_3_loss * FLAGS.level_3_weight + \
      level_32_loss * FLAGS.level_32_weight + \
      level_21_loss * FLAGS.level_21_weight

  # optimizer
  with tf.name_scope('optimizer'):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      tvars = tf.trainable_variables()
      with tf.name_scope('weight_decay'):
        regularizer = tf.add_n([tf.nn.l2_loss(v) for v in tvars])
      with tf.name_scope('learning_rate'):
        global_step = tf.Variable(0, trainable=False)
        if FLAGS.decay_policy =='step':
          boundaries = [int(max_iter*0.5), int(max_iter*0.75)]
          values = [i*FLAGS.learning_rate for i in [1, 0.1, 0.01]]
          lr = tf.train.piecewise_constant(global_step, boundaries, values)
        elif FLAGS.decay_policy == 'poly':
          lr = tf.train.polynomial_decay(FLAGS.learning_rate, global_step,
              int(max_iter*0.75), FLAGS.learning_rate*0.01, power=2)
        else:
          lr = FLAGS.learning_rate

      if FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(lr)
      else:
        optimizer = tf.train.MomentumOptimizer(lr, 0.9)

      solver = optimizer.minimize(train_loss + regularizer * FLAGS.weight_decay, var_list=tvars, global_step=global_step)

  with tf.name_scope('train_summary'):
    with tf.name_scope('loss'):
      summary_train_loss = tf.summary.scalar('train_loss', train_loss)
      summary_regularizer = tf.summary.scalar('regularizer', regularizer)
      summary_level_1_loss = tf.summary.scalar('level_1_loss', level_1_loss)
      summary_level_2_loss = tf.summary.scalar('level_2_loss', level_2_loss)
      summary_level_3_loss = tf.summary.scalar('level_3_loss', level_3_loss)
      summary_level_32_loss = tf.summary.scalar('level_32_loss', level_32_loss)
      summary_level_21_loss = tf.summary.scalar('level_21_loss', level_21_loss)
      summary_su_seg_loss_1 = tf.summary.scalar('su_seg_loss_1', su_seg_loss_1)
      summary_su_seg_loss_2 = tf.summary.scalar('su_seg_loss_2', su_seg_loss_2)
      summary_su_seg_loss_3 = tf.summary.scalar('su_seg_loss_3', su_seg_loss_3)
      summary_point_consistency_loss_1 = tf.summary.scalar('point_consistency_loss_1', point_consistency_loss_1)
      summary_point_consistency_loss_2 = tf.summary.scalar('point_consistency_loss_2', point_consistency_loss_2)
      summary_point_consistency_loss_3 = tf.summary.scalar('point_consistency_loss_3', point_consistency_loss_3)
      summary_part_consistency_loss_1 = tf.summary.scalar('part_consistency_loss_1', part_consistency_loss_1)
      summary_part_consistency_loss_2 = tf.summary.scalar('part_consistency_loss_2', part_consistency_loss_2)
      summary_part_consistency_loss_3 = tf.summary.scalar('part_consistency_loss_3', part_consistency_loss_3)
      summary_hierarchy_point_consistency_loss_32 = tf.summary.scalar('hierarchy_point_consistency_loss_32', hierarchy_point_consistency_loss_32)
      summary_hierarchy_point_consistency_loss_21 = tf.summary.scalar('hierarchy_point_consistency_loss_21', hierarchy_point_consistency_loss_21)
    with tf.name_scope('point'):
      with tf.name_scope('self'):
        summary_su_seg_accuracy_1 = tf.summary.scalar('su_seg_accuracy_1', su_seg_accuracy_1)
        summary_su_seg_accuracy_2 = tf.summary.scalar('su_seg_accuracy_2', su_seg_accuracy_2)
        summary_su_seg_accuracy_3 = tf.summary.scalar('su_seg_accuracy_3', su_seg_accuracy_3)
        summary_un_seg_loss_1 = tf.summary.scalar('un_seg_loss_1', un_seg_loss_1)
        summary_un_seg_loss_2 = tf.summary.scalar('un_seg_loss_2', un_seg_loss_2)
        summary_un_seg_loss_3 = tf.summary.scalar('un_seg_loss_3', un_seg_loss_3)
        summary_un_seg_accuracy_1 = tf.summary.scalar('un_seg_accuracy_1', un_seg_accuracy_1)
        summary_un_seg_accuracy_2 = tf.summary.scalar('un_seg_accuracy_2', un_seg_accuracy_2)
        summary_un_seg_accuracy_3 = tf.summary.scalar('un_seg_accuracy_3', un_seg_accuracy_3)
      with tf.name_scope('cst'):
        summary_point_consistency_accuracy_1 = tf.summary.scalar('point_consistency_accuracy_1', point_consistency_accuracy_1)
        summary_point_consistency_accuracy_2 = tf.summary.scalar('point_consistency_accuracy_2', point_consistency_accuracy_2)
        summary_point_consistency_accuracy_3 = tf.summary.scalar('point_consistency_accuracy_3', point_consistency_accuracy_3)
    with tf.name_scope('part'):
      with tf.name_scope('cst'):
        summary_part_consistency_accuracy_1 = tf.summary.scalar('part_consistency_accuracy_1', part_consistency_accuracy_1)
        summary_part_consistency_accuracy_2 = tf.summary.scalar('part_consistency_accuracy_2', part_consistency_accuracy_2)
        summary_part_consistency_accuracy_3 = tf.summary.scalar('part_consistency_accuracy_3', part_consistency_accuracy_3)

    with tf.name_scope('misc'):
      summary_lr_scheme = tf.summary.scalar('learning_rate', lr)
      summary_points_flag = tf.summary.scalar('points_flag', tf.reduce_mean(points_flag))
    train_merged = tf.summary.merge([
        summary_train_loss,
        summary_regularizer,
        summary_level_1_loss,
        summary_level_2_loss,
        summary_level_3_loss,
        summary_level_32_loss,
        summary_level_21_loss,
        summary_su_seg_loss_1,
        summary_su_seg_loss_2,
        summary_su_seg_loss_3,
        summary_point_consistency_loss_1,
        summary_point_consistency_loss_2,
        summary_point_consistency_loss_3,
        summary_part_consistency_loss_1,
        summary_part_consistency_loss_2,
        summary_part_consistency_loss_3,
        summary_hierarchy_point_consistency_loss_32,
        summary_hierarchy_point_consistency_loss_21,
        summary_su_seg_accuracy_1,
        summary_su_seg_accuracy_2,
        summary_su_seg_accuracy_3,
        summary_un_seg_loss_1,
        summary_un_seg_loss_2,
        summary_un_seg_loss_3,
        summary_un_seg_accuracy_1,
        summary_un_seg_accuracy_2,
        summary_un_seg_accuracy_3,
        summary_point_consistency_accuracy_1,
        summary_point_consistency_accuracy_2,
        summary_point_consistency_accuracy_3,
        summary_part_consistency_accuracy_1,
        summary_part_consistency_accuracy_2,
        summary_part_consistency_accuracy_3,
        summary_lr_scheme,
        summary_points_flag
    ])

  return train_merged, solver


def test_network(test_data, visual=True):
  [
      octree,                           # string
      node_position,                    # [n_point, 4]
      point_gt_part_index_1,            # [n_point]
      point_gt_part_index_2,            # [n_point]
      point_gt_part_index_3,            # [n_point]
      point_batch_index,                # [n_point]
  ] = get_test_input_data(test_data, FLAGS.test_batch_size, depth=6)


  # for segmentation
  with tf.variable_scope('seg'):
    point_feature = backbone(octree, node_position, training=False, reuse=True, nempty=FLAGS.nempty) # [1, C, n_point, 1]
    point_predict_logits_1, point_predict_prob_1, _ = seg_header(
        point_feature, n_part_1, training=False, reuse=True, suffix='_level_1') # [n_point, n_part_1], [n_point, n_part_1]
    point_predict_logits_2, point_predict_prob_2, _ = seg_header(
        point_feature, n_part_2, training=False, reuse=True, suffix='_level_2') # [n_point, n_part_2], [n_point, n_part_2]
    point_predict_logits_3, point_predict_prob_3, _ = seg_header(
        point_feature, n_part_3, training=False, reuse=True, suffix='_level_3') # [n_point, n_part_3], [n_point, n_part_3]

  # segmentation loss
  with tf.name_scope('seg_loss'):
    with tf.name_scope('level1'):
      seg_loss_1 = compute_segmentation_loss(
          point_predict_logits_1, point_gt_part_index_1, point_batch_index, n_part_1, delete_0=FLAGS.delete_0) # scalar
      seg_accuracy_1, point_predict_part_index_1 = compute_segmentation_accuracy(
          point_predict_prob_1, point_gt_part_index_1, point_batch_index, delete_0=FLAGS.delete_0) # scalar, [n_point]
    with tf.name_scope('level2'):
      seg_loss_2 = compute_segmentation_loss(
          point_predict_logits_2, point_gt_part_index_2, point_batch_index, n_part_2, delete_0=FLAGS.delete_0) # scalar
      seg_accuracy_2, point_predict_part_index_2 = compute_segmentation_accuracy(
          point_predict_prob_2, point_gt_part_index_2, point_batch_index, delete_0=FLAGS.delete_0) # scalar, [n_point]
    with tf.name_scope('level3'):
      seg_loss_3 = compute_segmentation_loss(
          point_predict_logits_3, point_gt_part_index_3, point_batch_index, n_part_3, delete_0=FLAGS.delete_0) # scalar
      seg_accuracy_3, point_predict_part_index_3 = compute_segmentation_accuracy(
          point_predict_prob_3, point_gt_part_index_3, point_batch_index, delete_0=FLAGS.delete_0) # scalar, [n_point]

  # part mask
  with tf.name_scope('part_mask_accuracy'):
    with tf.name_scope('level_1'):
      part_mask_accuracy_1, part_predict_mask_1, part_gt_mask_1 = compute_part_predict_accuracy(
          point_predict_part_index_1, point_gt_part_index_1, point_batch_index, n_part_1, delete_0=FLAGS.delete_0) # scalar, [bs, n_part_1], [bs, n_part_1]
    with tf.name_scope('level_2'):
      part_mask_accuracy_2, part_predict_mask_2, part_gt_mask_2 = compute_part_predict_accuracy(
          point_predict_part_index_2, point_gt_part_index_2, point_batch_index, n_part_2, delete_0=FLAGS.delete_0) # scalar, [bs, n_part_2], [bs, n_part_2]
    with tf.name_scope('level_3'):
      part_mask_accuracy_3, part_predict_mask_3, part_gt_mask_3 = compute_part_predict_accuracy(
          point_predict_part_index_3, point_gt_part_index_3, point_batch_index, n_part_3, delete_0=FLAGS.delete_0) # scalar, [bs, n_part_3], [bs, n_part_3]

  # total loss
  test_loss = FLAGS.level_1_weight*seg_loss_1 + FLAGS.level_2_weight*seg_loss_2 + FLAGS.level_3_weight*seg_loss_3

  with tf.name_scope('test_summary'):
    average_test_loss = tf.placeholder(tf.float32)
    average_seg_loss_1 = tf.placeholder(tf.float32)
    average_seg_accuracy_1 = tf.placeholder(tf.float32)
    average_part_mask_accuracy_1 = tf.placeholder(tf.float32)
    average_structure_accuracy_1 = tf.placeholder(tf.float32)
    average_part_diff_1 = tf.placeholder(tf.float32)
    average_miou_v1_1 = tf.placeholder(tf.float32)
    average_miou_v2_1 = tf.placeholder(tf.float32)
    average_miou_v3_1 = tf.placeholder(tf.float32)
    average_seg_loss_2 = tf.placeholder(tf.float32)
    average_seg_accuracy_2 = tf.placeholder(tf.float32)
    average_part_mask_accuracy_2 = tf.placeholder(tf.float32)
    average_structure_accuracy_2 = tf.placeholder(tf.float32)
    average_part_diff_2 = tf.placeholder(tf.float32)
    average_miou_v1_2 = tf.placeholder(tf.float32)
    average_miou_v2_2 = tf.placeholder(tf.float32)
    average_miou_v3_2 = tf.placeholder(tf.float32)
    average_seg_loss_3 = tf.placeholder(tf.float32)
    average_seg_accuracy_3 = tf.placeholder(tf.float32)
    average_part_mask_accuracy_3 = tf.placeholder(tf.float32)
    average_structure_accuracy_3 = tf.placeholder(tf.float32)
    average_part_diff_3 = tf.placeholder(tf.float32)
    average_miou_v1_3 = tf.placeholder(tf.float32)
    average_miou_v2_3 = tf.placeholder(tf.float32)
    average_miou_v3_3 = tf.placeholder(tf.float32)
    summary_test_loss = tf.summary.scalar('test_loss', average_test_loss)
    summary_seg_loss_1 = tf.summary.scalar('seg_loss_1', average_seg_loss_1)
    summary_seg_accuracy_1 = tf.summary.scalar('seg_accuracy_1', average_seg_accuracy_1)
    summary_part_mask_accuracy_1 = tf.summary.scalar('part_mask_accuracy_1', average_part_mask_accuracy_1)
    summary_structure_accuracy_1 = tf.summary.scalar('structure_accuracy_1', average_structure_accuracy_1)
    summary_part_diff_1 = tf.summary.scalar('part_diff_1', average_part_diff_1)
    summary_miou_v1_1 = tf.summary.scalar('miou_v1_1', average_miou_v1_1)
    summary_miou_v2_1 = tf.summary.scalar('miou_v2_1', average_miou_v2_1)
    summary_miou_v3_1 = tf.summary.scalar('miou_v3_1', average_miou_v3_1)
    summary_seg_loss_2 = tf.summary.scalar('seg_loss_2', average_seg_loss_2)
    summary_seg_accuracy_2 = tf.summary.scalar('seg_accuracy_2', average_seg_accuracy_2)
    summary_part_mask_accuracy_2 = tf.summary.scalar('part_mask_accuracy_2', average_part_mask_accuracy_2)
    summary_structure_accuracy_2 = tf.summary.scalar('structure_accuracy_2', average_structure_accuracy_2)
    summary_part_diff_2 = tf.summary.scalar('part_diff_2', average_part_diff_2)
    summary_miou_v1_2 = tf.summary.scalar('miou_v1_2', average_miou_v1_2)
    summary_miou_v2_2 = tf.summary.scalar('miou_v2_2', average_miou_v2_2)
    summary_miou_v3_2 = tf.summary.scalar('miou_v3_2', average_miou_v3_2)
    summary_seg_loss_3 = tf.summary.scalar('seg_loss_3', average_seg_loss_3)
    summary_seg_accuracy_3 = tf.summary.scalar('seg_accuracy_3', average_seg_accuracy_3)
    summary_part_mask_accuracy_3 = tf.summary.scalar('part_mask_accuracy_3', average_part_mask_accuracy_3)
    summary_structure_accuracy_3 = tf.summary.scalar('structure_accuracy_3', average_structure_accuracy_3)
    summary_part_diff_3 = tf.summary.scalar('part_diff_3', average_part_diff_3)
    summary_miou_v1_3 = tf.summary.scalar('miou_v1_3', average_miou_v1_3)
    summary_miou_v2_3 = tf.summary.scalar('miou_v2_3', average_miou_v2_3)
    summary_miou_v3_3 = tf.summary.scalar('miou_v3_3', average_miou_v3_3)
    test_merged = tf.summary.merge([
        summary_test_loss,
        summary_seg_loss_1,
        summary_seg_accuracy_1,
        summary_part_mask_accuracy_1,
        summary_structure_accuracy_1,
        summary_part_diff_1,
        summary_miou_v1_1,
        summary_miou_v2_1,
        summary_miou_v3_1,
        summary_seg_loss_2,
        summary_seg_accuracy_2,
        summary_part_mask_accuracy_2,
        summary_structure_accuracy_2,
        summary_part_diff_2,
        summary_miou_v1_2,
        summary_miou_v2_2,
        summary_miou_v3_2,
        summary_seg_loss_3,
        summary_seg_accuracy_3,
        summary_part_mask_accuracy_3,
        summary_structure_accuracy_3,
        summary_part_diff_3,
        summary_miou_v1_3,
        summary_miou_v2_3,
        summary_miou_v3_3,
    ])
  return_list = [
      test_merged,
      average_test_loss,
      average_seg_loss_1,
      average_seg_accuracy_1,
      average_part_mask_accuracy_1,
      average_structure_accuracy_1,
      average_part_diff_1,
      average_miou_v1_1,
      average_miou_v2_1,
      average_miou_v3_1,
      average_seg_loss_2,
      average_seg_accuracy_2,
      average_part_mask_accuracy_2,
      average_structure_accuracy_2,
      average_part_diff_2,
      average_miou_v1_2,
      average_miou_v2_2,
      average_miou_v3_2,
      average_seg_loss_3,
      average_seg_accuracy_3,
      average_part_mask_accuracy_3,
      average_structure_accuracy_3,
      average_part_diff_3,
      average_miou_v1_3,
      average_miou_v2_3,
      average_miou_v3_3,
      test_loss,
      seg_loss_1,
      seg_accuracy_1,
      part_mask_accuracy_1,
      point_gt_part_index_1,
      point_predict_part_index_1,
      point_predict_prob_1,
      part_gt_mask_1,
      part_predict_mask_1,
      seg_loss_2,
      seg_accuracy_2,
      part_mask_accuracy_2,
      point_gt_part_index_2,
      point_predict_part_index_2,
      point_predict_prob_2,
      part_gt_mask_2,
      part_predict_mask_2,
      seg_loss_3,
      seg_accuracy_3,
      part_mask_accuracy_3,
      point_gt_part_index_3,
      point_predict_part_index_3,
      point_predict_prob_3,
      part_gt_mask_3,
      part_predict_mask_3,
      node_position
  ]
  if visual:
    return_list = [
        node_position,
        point_gt_part_index_1,
        point_predict_part_index_1,
        point_predict_prob_1,
        point_gt_part_index_2,
        point_predict_part_index_2,
        point_predict_prob_2,
        point_gt_part_index_3,
        point_predict_part_index_3,
        point_predict_prob_3
    ]

  return return_list


def main(argv=None):

  train_summary, solver = train_network()

  # visual test
  [
      visual_node_position,
      visual_point_gt_part_index_1,
      visual_point_predict_part_index_1,
      visual_point_predict_prob_1,
      visual_point_gt_part_index_2,
      visual_point_predict_part_index_2,
      visual_point_predict_prob_2,
      visual_point_gt_part_index_3,
      visual_point_predict_part_index_3,
      visual_point_predict_prob_3
  ] = test_network(FLAGS.test_data_visual, visual=True)

  # non-visual test
  [
      test_summary,
      average_test_loss,
      average_seg_loss_1,
      average_seg_accuracy_1,
      average_part_mask_accuracy_1,
      average_structure_accuracy_1,
      average_part_diff_1,
      average_miou_v1_1,
      average_miou_v2_1,
      average_miou_v3_1,
      average_seg_loss_2,
      average_seg_accuracy_2,
      average_part_mask_accuracy_2,
      average_structure_accuracy_2,
      average_part_diff_2,
      average_miou_v1_2,
      average_miou_v2_2,
      average_miou_v3_2,
      average_seg_loss_3,
      average_seg_accuracy_3,
      average_part_mask_accuracy_3,
      average_structure_accuracy_3,
      average_part_diff_3,
      average_miou_v1_3,
      average_miou_v2_3,
      average_miou_v3_3,
      test_loss,
      seg_loss_1,
      seg_accuracy_1,
      part_mask_accuracy_1,
      point_gt_part_index_1,
      point_predict_part_index_1,
      point_predict_prob_1,
      part_gt_mask_1,
      part_predict_mask_1,
      seg_loss_2,
      seg_accuracy_2,
      part_mask_accuracy_2,
      point_gt_part_index_2,
      point_predict_part_index_2,
      point_predict_prob_2,
      part_gt_mask_2,
      part_predict_mask_2,
      seg_loss_3,
      seg_accuracy_3,
      part_mask_accuracy_3,
      point_gt_part_index_3,
      point_predict_part_index_3,
      point_predict_prob_3,
      part_gt_mask_3,
      part_predict_mask_3,
      node_position
  ] = test_network(FLAGS.test_data, visual=False)

  # checkpoint
  ckpt = tf.train.latest_checkpoint(FLAGS.ckpt)
  start_iters = 0 if not ckpt else int(ckpt[ckpt.find('iter') + 4:-5]) + 1

  # saver
  allvars = tf.all_variables()

  tf_saver = tf.train.Saver(var_list=allvars, max_to_keep=2)
  if ckpt:
    assert(os.path.exists(FLAGS.ckpt))
    tf_restore_saver = tf.train.Saver(var_list=allvars, max_to_keep=2)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    # initialize
    init = tf.global_variables_initializer()
    sess.run(init)

    if ckpt:
      tf_restore_saver.restore(sess, ckpt)

    obj_dir = os.path.join('obj', FLAGS.cache_folder)
    if not os.path.exists(obj_dir): os.makedirs(obj_dir)

    if FLAGS.phase == 'train':
      # tf summary
      summary_writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)

      # start training
      for i in tqdm(range(start_iters, max_iter + 1)):

        if i % FLAGS.test_every_iter == 0 and i != 0:
          # evaluate on whole test data
          avg_test_loss = 0
          avg_seg_loss_1 = 0
          avg_seg_accuracy_1 = 0
          avg_part_mask_accuracy_1 = 0
          all_point_gt_part_index_1 = np.empty([test_iter*n_test_point], dtype=int)
          all_point_predict_part_index_1 = np.empty([test_iter*n_test_point], dtype=int)
          all_part_gt_mask_1 = np.empty([test_iter, n_part_1], dtype=int)
          all_part_predict_mask_1 = np.empty([test_iter, n_part_1], dtype=int)
          avg_seg_loss_2 = 0
          avg_seg_accuracy_2 = 0
          avg_part_mask_accuracy_2 = 0
          all_point_gt_part_index_2 = np.empty([test_iter*n_test_point], dtype=int)
          all_point_predict_part_index_2 = np.empty([test_iter*n_test_point], dtype=int)
          all_part_gt_mask_2 = np.empty([test_iter, n_part_2], dtype=int)
          all_part_predict_mask_2 = np.empty([test_iter, n_part_2], dtype=int)
          avg_seg_loss_3 = 0
          avg_seg_accuracy_3 = 0
          avg_part_mask_accuracy_3 = 0
          all_point_gt_part_index_3 = np.empty([test_iter*n_test_point], dtype=int)
          all_point_predict_part_index_3 = np.empty([test_iter*n_test_point], dtype=int)
          all_part_gt_mask_3 = np.empty([test_iter, n_part_3], dtype=int)
          all_part_predict_mask_3 = np.empty([test_iter, n_part_3], dtype=int)
          all_point_shape_index = np.empty([test_iter*n_test_point], dtype=int)
          n_point_count = 0
          for it in tqdm(range(test_iter)):
            [
                test_loss_value,
                seg_loss_1_value,
                seg_accuracy_1_value,
                part_mask_accuracy_1_value,
                point_gt_part_index_1_value,
                point_predict_part_index_1_value,
                part_gt_mask_1_value,
                part_predict_mask_1_value,
                seg_loss_2_value,
                seg_accuracy_2_value,
                part_mask_accuracy_2_value,
                point_gt_part_index_2_value,
                point_predict_part_index_2_value,
                part_gt_mask_2_value,
                part_predict_mask_2_value,
                seg_loss_3_value,
                seg_accuracy_3_value,
                part_mask_accuracy_3_value,
                point_gt_part_index_3_value,
                point_predict_part_index_3_value,
                part_gt_mask_3_value,
                part_predict_mask_3_value,
            ] = sess.run(
                [
                    test_loss,
                    seg_loss_1,
                    seg_accuracy_1,
                    part_mask_accuracy_1,
                    point_gt_part_index_1,
                    point_predict_part_index_1,
                    part_gt_mask_1,
                    part_predict_mask_1,
                    seg_loss_2,
                    seg_accuracy_2,
                    part_mask_accuracy_2,
                    point_gt_part_index_2,
                    point_predict_part_index_2,
                    part_gt_mask_2,
                    part_predict_mask_2,
                    seg_loss_3,
                    seg_accuracy_3,
                    part_mask_accuracy_3,
                    point_gt_part_index_3,
                    point_predict_part_index_3,
                    part_gt_mask_3,
                    part_predict_mask_3,
                ]
            )

            n_shape_point = point_gt_part_index_1_value.size
            assert n_point_count + n_shape_point <= test_iter*n_test_point, 'Test point number {} > {}={}*{}'.format(n_point_count + n_shape_point, test_iter*n_test_point, test_iter, n_test_point)
            avg_test_loss += test_loss_value
            avg_seg_loss_1 += seg_loss_1_value
            avg_seg_accuracy_1 += seg_accuracy_1_value
            avg_part_mask_accuracy_1 += part_mask_accuracy_1_value
            all_point_gt_part_index_1[n_point_count:n_point_count+n_shape_point] = point_gt_part_index_1_value
            all_point_predict_part_index_1[n_point_count:n_point_count+n_shape_point] = point_predict_part_index_1_value
            all_part_gt_mask_1[it] = part_gt_mask_1_value
            all_part_predict_mask_1[it] = part_predict_mask_1_value
            avg_seg_loss_2 += seg_loss_2_value
            avg_seg_accuracy_2 += seg_accuracy_2_value
            avg_part_mask_accuracy_2 += part_mask_accuracy_2_value
            all_point_gt_part_index_2[n_point_count:n_point_count+n_shape_point] = point_gt_part_index_2_value
            all_point_predict_part_index_2[n_point_count:n_point_count+n_shape_point] = point_predict_part_index_2_value
            all_part_gt_mask_2[it] = part_gt_mask_2_value
            all_part_predict_mask_2[it] = part_predict_mask_2_value
            avg_seg_loss_3 += seg_loss_3_value
            avg_seg_accuracy_3 += seg_accuracy_3_value
            avg_part_mask_accuracy_3 += part_mask_accuracy_3_value
            all_point_gt_part_index_3[n_point_count:n_point_count+n_shape_point] = point_gt_part_index_3_value
            all_point_predict_part_index_3[n_point_count:n_point_count+n_shape_point] = point_predict_part_index_3_value
            all_part_gt_mask_3[it] = part_gt_mask_3_value
            all_part_predict_mask_3[it] = part_predict_mask_3_value
            all_point_shape_index[n_point_count:n_point_count+n_shape_point] = it
            n_point_count += n_shape_point
          all_point_gt_part_index_1 = all_point_gt_part_index_1[:n_point_count]
          all_point_predict_part_index_1 = all_point_predict_part_index_1[:n_point_count]
          all_point_gt_part_index_2 = all_point_gt_part_index_2[:n_point_count]
          all_point_predict_part_index_2 = all_point_predict_part_index_2[:n_point_count]
          all_point_gt_part_index_3 = all_point_gt_part_index_3[:n_point_count]
          all_point_predict_part_index_3 = all_point_predict_part_index_3[:n_point_count]
          all_point_shape_index = all_point_shape_index[:n_point_count]
          avg_test_loss /= test_iter
          avg_seg_loss_1 /= test_iter
          avg_seg_accuracy_1 /= test_iter
          avg_part_mask_accuracy_1 /= test_iter
          miou_v1_1, iou_v1_1_value = compute_iou_v1(all_point_predict_part_index_1, all_point_gt_part_index_1, all_point_shape_index, n_part_1, delete_0=FLAGS.delete_0)
          miou_v2_1, iou_v2_1_value = compute_iou_v2(all_point_predict_part_index_1, all_point_gt_part_index_1, n_part_1, delete_0=FLAGS.delete_0)
          miou_v3_1, iou_v3_1_value = compute_iou_v3(all_point_predict_part_index_1, all_point_gt_part_index_1, all_point_shape_index, n_part_1, delete_0=FLAGS.delete_0)
          structure_accuracy_1, part_diff_1 = compute_structure_accuracy(all_part_predict_mask_1, all_part_gt_mask_1, delete_0=FLAGS.delete_0)
          avg_seg_loss_2 /= test_iter
          avg_seg_accuracy_2 /= test_iter
          avg_part_mask_accuracy_2 /= test_iter
          miou_v1_2, iou_v1_2_value = compute_iou_v1(all_point_predict_part_index_2, all_point_gt_part_index_2, all_point_shape_index, n_part_2, delete_0=FLAGS.delete_0)
          miou_v2_2, iou_v2_2_value = compute_iou_v2(all_point_predict_part_index_2, all_point_gt_part_index_2, n_part_2, delete_0=FLAGS.delete_0)
          miou_v3_2, iou_v3_2_value = compute_iou_v3(all_point_predict_part_index_2, all_point_gt_part_index_2, all_point_shape_index, n_part_2, delete_0=FLAGS.delete_0)
          structure_accuracy_2, part_diff_2 = compute_structure_accuracy(all_part_predict_mask_2, all_part_gt_mask_2, delete_0=FLAGS.delete_0)
          avg_seg_loss_3 /= test_iter
          avg_seg_accuracy_3 /= test_iter
          avg_part_mask_accuracy_3 /= test_iter
          miou_v1_3, iou_v1_3_value = compute_iou_v1(all_point_predict_part_index_3, all_point_gt_part_index_3, all_point_shape_index, n_part_3, delete_0=FLAGS.delete_0)
          miou_v2_3, iou_v2_3_value = compute_iou_v2(all_point_predict_part_index_3, all_point_gt_part_index_3, n_part_3, delete_0=FLAGS.delete_0)
          miou_v3_3, iou_v3_3_value = compute_iou_v3(all_point_predict_part_index_3, all_point_gt_part_index_3, all_point_shape_index, n_part_3, delete_0=FLAGS.delete_0)
          structure_accuracy_3, part_diff_3 = compute_structure_accuracy(all_part_predict_mask_3, all_part_gt_mask_3, delete_0=FLAGS.delete_0)
          summary = sess.run(test_summary,
              feed_dict={
                  average_test_loss: avg_test_loss,
                  average_seg_loss_1: avg_seg_loss_1,
                  average_seg_accuracy_1: avg_seg_accuracy_1,
                  average_part_mask_accuracy_1: avg_part_mask_accuracy_1,
                  average_structure_accuracy_1: structure_accuracy_1,
                  average_part_diff_1: part_diff_1,
                  average_miou_v1_1: miou_v1_1,
                  average_miou_v2_1: miou_v2_1,
                  average_miou_v3_1: miou_v3_1,
                  average_seg_loss_2: avg_seg_loss_2,
                  average_seg_accuracy_2: avg_seg_accuracy_2,
                  average_part_mask_accuracy_2: avg_part_mask_accuracy_2,
                  average_structure_accuracy_2: structure_accuracy_2,
                  average_part_diff_2: part_diff_2,
                  average_miou_v1_2: miou_v1_2,
                  average_miou_v2_2: miou_v2_2,
                  average_miou_v3_2: miou_v3_2,
                  average_seg_loss_3: avg_seg_loss_3,
                  average_seg_accuracy_3: avg_seg_accuracy_3,
                  average_part_mask_accuracy_3: avg_part_mask_accuracy_3,
                  average_structure_accuracy_3: structure_accuracy_3,
                  average_part_diff_3: part_diff_3,
                  average_miou_v1_3: miou_v1_3,
                  average_miou_v2_3: miou_v2_3,
                  average_miou_v3_3: miou_v3_3,
              })
          summary_writer.add_summary(summary, i)
          tf_saver.save(sess, os.path.join(FLAGS.logdir, 'model/iter{:06d}.ckpt'.format(i)))

          result_string = '\nIteration {}:\nLevel 1:\n miou v1     : {:5.2f}\n miou v2     : {:5.2f}\n miou v3     : {:5.2f}\n seg accu    : {:6.4f}\n part diff   :  {:6.4f}\n struct accu :  {:6.4f}\n part accu   :  {:6.4f}\n seg loss    :  {:6.4f}\n\nLevel 2:\n miou v1     : {:5.2f}\n miou v2     : {:5.2f}\n miou v3     : {:5.2f}\n seg accu    : {:6.4f}\n part diff   :  {:6.4f}\n struct accu :  {:6.4f}\n part accu   :  {:6.4f}\n seg loss    :  {:6.4f}\n\nLevel 3:\n miou v1     : {:5.2f}\n miou v2     : {:5.2f}\n miou v3     : {:5.2f}\n seg accu    : {:6.4f}\n part diff   :  {:6.4f}\n struct accu :  {:6.4f}\n part accu   :  {:6.4f}\n seg loss    :  {:6.4f}\n'.format(i, miou_v1_1*100, miou_v2_1*100, miou_v3_1*100, avg_seg_accuracy_1, part_diff_1, structure_accuracy_1, avg_part_mask_accuracy_1, avg_seg_loss_1, miou_v1_2*100, miou_v2_2*100, miou_v3_2*100, avg_seg_accuracy_2, part_diff_2, structure_accuracy_2, avg_part_mask_accuracy_2, avg_seg_loss_2, miou_v1_3*100, miou_v2_3*100, miou_v3_3*100, avg_seg_accuracy_3, part_diff_3, structure_accuracy_3, avg_part_mask_accuracy_3, avg_seg_loss_3)
          print(result_string); sys.stdout.flush()

          # visualize on typical shapes
          if (i % FLAGS.disp_every_n_steps == 0):
            for it in range(test_iter_visual):
              [
                  visual_node_position_value,
                  visual_point_gt_part_index_1_value,
                  visual_point_predict_part_index_1_value,
                  visual_point_predict_prob_1_value,
                  visual_point_gt_part_index_2_value,
                  visual_point_predict_part_index_2_value,
                  visual_point_predict_prob_2_value,
                  visual_point_gt_part_index_3_value,
                  visual_point_predict_part_index_3_value,
                  visual_point_predict_prob_3_value
              ] = sess.run(
                  [
                      visual_node_position,
                      visual_point_gt_part_index_1,
                      visual_point_predict_part_index_1,
                      visual_point_predict_prob_1,
                      visual_point_gt_part_index_2,
                      visual_point_predict_part_index_2,
                      visual_point_predict_prob_2,
                      visual_point_gt_part_index_3,
                      visual_point_predict_part_index_3,
                      visual_point_predict_prob_3
                   ]
              )

              pc_filename = os.path.join(obj_dir, 'pc_L1_gt_{:06d}_{:04d}.obj'.format(i, it))
              vis_pointcloud.save_points(visual_node_position_value,
                  visual_point_gt_part_index_1_value.flatten(), pc_filename, depth=5)

              pc_filename = os.path.join(obj_dir, 'pc_L1_perd_{:06d}_{:04d}.obj'.format(i, it))
              vis_pointcloud.save_points(visual_node_position_value,
                  visual_point_predict_part_index_1_value.flatten(), pc_filename, depth=5)

              pc_filename = os.path.join(obj_dir, 'pc_L1_err_{:06d}_{:04d}.obj'.format(i, it))
              vis_pointcloud.save_points(visual_node_position_value,
                  (visual_point_predict_part_index_1_value.flatten()!=visual_point_gt_part_index_1_value.flatten())*4,
                  pc_filename, depth=5, squantial_color=True)

              pc_filename = os.path.join(obj_dir, 'pc_L2_gt_{:06d}_{:04d}.obj'.format(i, it))
              vis_pointcloud.save_points(visual_node_position_value,
                  visual_point_gt_part_index_2_value.flatten(), pc_filename, depth=5)

              pc_filename = os.path.join(obj_dir, 'pc_L2_perd_{:06d}_{:04d}.obj'.format(i, it))
              vis_pointcloud.save_points(visual_node_position_value,
                  visual_point_predict_part_index_2_value.flatten(), pc_filename, depth=5)

              pc_filename = os.path.join(obj_dir, 'pc_L2_err_{:06d}_{:04d}.obj'.format(i, it))
              vis_pointcloud.save_points(visual_node_position_value,
                  (visual_point_predict_part_index_2_value.flatten()!=visual_point_gt_part_index_2_value.flatten())*4,
                  pc_filename, depth=5, squantial_color=True)

              pc_filename = os.path.join(obj_dir, 'pc_L3_gt_{:06d}_{:04d}.obj'.format(i, it))
              vis_pointcloud.save_points(visual_node_position_value,
                  visual_point_gt_part_index_3_value.flatten(), pc_filename, depth=5)

              pc_filename = os.path.join(obj_dir, 'pc_L3_perd_{:06d}_{:04d}.obj'.format(i, it))
              vis_pointcloud.save_points(visual_node_position_value,
                  visual_point_predict_part_index_3_value.flatten(), pc_filename, depth=5)

              pc_filename = os.path.join(obj_dir, 'pc_L3_err_{:06d}_{:04d}.obj'.format(i, it))
              vis_pointcloud.save_points(visual_node_position_value,
                  (visual_point_predict_part_index_3_value.flatten()!=visual_point_gt_part_index_3_value.flatten())*4,
                  pc_filename, depth=5, squantial_color=True)

        summary, _ = sess.run([train_summary, solver])
        summary_writer.add_summary(summary, i)


    else:
      assert(FLAGS.phase == 'test')
      # run_time
      avg_test_loss = 0
      avg_seg_loss_1 = 0
      avg_seg_accuracy_1 = 0
      avg_part_mask_accuracy_1 = 0
      all_point_gt_part_index_1 = np.empty([test_iter*n_test_point], dtype=int)
      all_point_predict_part_index_1 = np.empty([test_iter*n_test_point], dtype=int)
      all_part_gt_mask_1 = np.empty([test_iter, n_part_1], dtype=int)
      all_part_predict_mask_1 = np.empty([test_iter, n_part_1], dtype=int)
      avg_seg_loss_2 = 0
      avg_seg_accuracy_2 = 0
      avg_part_mask_accuracy_2 = 0
      all_point_gt_part_index_2 = np.empty([test_iter*n_test_point], dtype=int)
      all_point_predict_part_index_2 = np.empty([test_iter*n_test_point], dtype=int)
      all_part_gt_mask_2 = np.empty([test_iter, n_part_2], dtype=int)
      all_part_predict_mask_2 = np.empty([test_iter, n_part_2], dtype=int)
      avg_seg_loss_3 = 0
      avg_seg_accuracy_3 = 0
      avg_part_mask_accuracy_3 = 0
      all_point_gt_part_index_3 = np.empty([test_iter*n_test_point], dtype=int)
      all_point_predict_part_index_3 = np.empty([test_iter*n_test_point], dtype=int)
      all_part_gt_mask_3 = np.empty([test_iter, n_part_3], dtype=int)
      all_part_predict_mask_3 = np.empty([test_iter, n_part_3], dtype=int)
      all_node_position = np.empty([test_iter*n_test_point, 3], dtype=np.float32)
      all_point_shape_index = np.empty([test_iter*n_test_point], dtype=int)
      n_point_count = 0
      for it in tqdm(range(test_iter)):
        [
            test_loss_value,
            seg_loss_1_value,
            seg_accuracy_1_value,
            part_mask_accuracy_1_value,
            point_gt_part_index_1_value,
            point_predict_part_index_1_value,
            part_gt_mask_1_value,
            part_predict_mask_1_value,
            seg_loss_2_value,
            seg_accuracy_2_value,
            part_mask_accuracy_2_value,
            point_gt_part_index_2_value,
            point_predict_part_index_2_value,
            part_gt_mask_2_value,
            part_predict_mask_2_value,
            seg_loss_3_value,
            seg_accuracy_3_value,
            part_mask_accuracy_3_value,
            point_gt_part_index_3_value,
            point_predict_part_index_3_value,
            part_gt_mask_3_value,
            part_predict_mask_3_value,
            node_position_value,
        ] = sess.run(
            [
                test_loss,
                seg_loss_1,
                seg_accuracy_1,
                part_mask_accuracy_1,
                point_gt_part_index_1,
                point_predict_part_index_1,
                part_gt_mask_1,
                part_predict_mask_1,
                seg_loss_2,
                seg_accuracy_2,
                part_mask_accuracy_2,
                point_gt_part_index_2,
                point_predict_part_index_2,
                part_gt_mask_2,
                part_predict_mask_2,
                seg_loss_3,
                seg_accuracy_3,
                part_mask_accuracy_3,
                point_gt_part_index_3,
                point_predict_part_index_3,
                part_gt_mask_3,
                part_predict_mask_3,
                node_position,
            ]
        )

        n_shape_point = point_gt_part_index_1_value.size
        assert n_point_count + n_shape_point <= test_iter*n_test_point, 'Test point number {} > {}={}*{}'.format(n_point_count + n_shape_point, test_iter*n_test_point, test_iter, n_test_point)
        avg_test_loss += test_loss_value
        avg_seg_loss_1 += seg_loss_1_value
        avg_seg_accuracy_1 += seg_accuracy_1_value
        avg_part_mask_accuracy_1 += part_mask_accuracy_1_value
        all_point_gt_part_index_1[n_point_count:n_point_count+n_shape_point] = point_gt_part_index_1_value
        all_point_predict_part_index_1[n_point_count:n_point_count+n_shape_point] = point_predict_part_index_1_value
        all_part_gt_mask_1[it] = part_gt_mask_1_value
        all_part_predict_mask_1[it] = part_predict_mask_1_value
        avg_seg_loss_2 += seg_loss_2_value
        avg_seg_accuracy_2 += seg_accuracy_2_value
        avg_part_mask_accuracy_2 += part_mask_accuracy_2_value
        all_point_gt_part_index_2[n_point_count:n_point_count+n_shape_point] = point_gt_part_index_2_value
        all_point_predict_part_index_2[n_point_count:n_point_count+n_shape_point] = point_predict_part_index_2_value
        all_part_gt_mask_2[it] = part_gt_mask_2_value
        all_part_predict_mask_2[it] = part_predict_mask_2_value
        avg_seg_loss_3 += seg_loss_3_value
        avg_seg_accuracy_3 += seg_accuracy_3_value
        avg_part_mask_accuracy_3 += part_mask_accuracy_3_value
        all_point_gt_part_index_3[n_point_count:n_point_count+n_shape_point] = point_gt_part_index_3_value
        all_point_predict_part_index_3[n_point_count:n_point_count+n_shape_point] = point_predict_part_index_3_value
        all_part_gt_mask_3[it] = part_gt_mask_3_value
        all_part_predict_mask_3[it] = part_predict_mask_3_value
        all_node_position[n_point_count:n_point_count+n_shape_point, :] = node_position_value[:, :3]
        all_point_shape_index[n_point_count:n_point_count+n_shape_point] = it
        n_point_count += n_shape_point

        if FLAGS.test_visual:

          pc_filename = os.path.join(obj_dir, 'pc_L1_gt_{:06d}_{:04d}.obj'.format(0, it))
          vis_pointcloud.save_points(node_position_value,
              point_gt_part_index_1_value.flatten(), pc_filename, depth=5)

          pc_filename = os.path.join(obj_dir, 'pc_L1_pred_{:06d}_{:04d}.obj'.format(0, it))
          vis_pointcloud.save_points(node_position_value,
              point_predict_part_index_1_value.flatten(), pc_filename, depth=5)

          pc_filename = os.path.join(obj_dir, 'pc_L1_err_{:06d}_{:04d}.obj'.format(0, it))
          vis_pointcloud.save_points(node_position_value,
              (point_predict_part_index_1_value.flatten()!=point_gt_part_index_1_value.flatten())*4,
              pc_filename, depth=5, squantial_color=True)

          pc_filename = os.path.join(obj_dir, 'pc_L2_gt_{:06d}_{:04d}.obj'.format(0, it))
          vis_pointcloud.save_points(node_position_value,
              point_gt_part_index_2_value.flatten(), pc_filename, depth=5)

          pc_filename = os.path.join(obj_dir, 'pc_L2_pred_{:06d}_{:04d}.obj'.format(0, it))
          vis_pointcloud.save_points(node_position_value,
              point_predict_part_index_2_value.flatten(), pc_filename, depth=5)

          pc_filename = os.path.join(obj_dir, 'pc_L2_err_{:06d}_{:04d}.obj'.format(0, it))
          vis_pointcloud.save_points(node_position_value,
              (point_predict_part_index_2_value.flatten()!=point_gt_part_index_2_value.flatten())*4,
              pc_filename, depth=5, squantial_color=True)

          pc_filename = os.path.join(obj_dir, 'pc_L3_gt_{:06d}_{:04d}.obj'.format(0, it))
          vis_pointcloud.save_points(node_position_value,
              point_gt_part_index_3_value.flatten(), pc_filename, depth=5)

          pc_filename = os.path.join(obj_dir, 'pc_L3_pred_{:06d}_{:04d}.obj'.format(0, it))
          vis_pointcloud.save_points(node_position_value,
              point_predict_part_index_3_value.flatten(), pc_filename, depth=5)

          pc_filename = os.path.join(obj_dir, 'pc_L3_err_{:06d}_{:04d}.obj'.format(0, it))
          vis_pointcloud.save_points(node_position_value,
              (point_predict_part_index_3_value.flatten()!=point_gt_part_index_3_value.flatten())*4,
              pc_filename, depth=5, squantial_color=True)

      all_point_gt_part_index_1 = all_point_gt_part_index_1[:n_point_count]
      all_point_predict_part_index_1 = all_point_predict_part_index_1[:n_point_count]
      all_point_gt_part_index_2 = all_point_gt_part_index_2[:n_point_count]
      all_point_predict_part_index_2 = all_point_predict_part_index_2[:n_point_count]
      all_point_gt_part_index_3 = all_point_gt_part_index_3[:n_point_count]
      all_point_predict_part_index_3 = all_point_predict_part_index_3[:n_point_count]
      all_node_position = all_node_position[:n_point_count, :]
      all_point_shape_index = all_point_shape_index[:n_point_count]
      avg_test_loss /= test_iter
      avg_seg_loss_1 /= test_iter
      avg_seg_accuracy_1 /= test_iter
      avg_part_mask_accuracy_1 /= test_iter
      miou_v1_1, iou_v1_1_value = compute_iou_v1(all_point_predict_part_index_1, all_point_gt_part_index_1, all_point_shape_index, n_part_1, delete_0=FLAGS.delete_0)
      miou_v2_1, iou_v2_1_value = compute_iou_v2(all_point_predict_part_index_1, all_point_gt_part_index_1, n_part_1, delete_0=FLAGS.delete_0)
      miou_v3_1, iou_v3_1_value = compute_iou_v3(all_point_predict_part_index_1, all_point_gt_part_index_1, all_point_shape_index, n_part_1, delete_0=FLAGS.delete_0)
      miou_v4_1 = compute_iou_v4(all_point_predict_part_index_1, all_point_gt_part_index_1, all_point_shape_index)
      structure_accuracy_1, part_diff_1 = compute_structure_accuracy(all_part_predict_mask_1, all_part_gt_mask_1, delete_0=FLAGS.delete_0)
      avg_seg_loss_2 /= test_iter
      avg_seg_accuracy_2 /= test_iter
      avg_part_mask_accuracy_2 /= test_iter
      miou_v1_2, iou_v1_2_value = compute_iou_v1(all_point_predict_part_index_2, all_point_gt_part_index_2, all_point_shape_index, n_part_2, delete_0=FLAGS.delete_0)
      miou_v2_2, iou_v2_2_value = compute_iou_v2(all_point_predict_part_index_2, all_point_gt_part_index_2, n_part_2, delete_0=FLAGS.delete_0)
      miou_v3_2, iou_v3_2_value = compute_iou_v3(all_point_predict_part_index_2, all_point_gt_part_index_2, all_point_shape_index, n_part_2, delete_0=FLAGS.delete_0)
      miou_v4_2 = compute_iou_v4(all_point_predict_part_index_2, all_point_gt_part_index_2, all_point_shape_index)
      structure_accuracy_2, part_diff_2 = compute_structure_accuracy(all_part_predict_mask_2, all_part_gt_mask_2, delete_0=FLAGS.delete_0)
      avg_seg_loss_3 /= test_iter
      avg_seg_accuracy_3 /= test_iter
      avg_part_mask_accuracy_3 /= test_iter
      miou_v1_3, iou_v1_3_value = compute_iou_v1(all_point_predict_part_index_3, all_point_gt_part_index_3, all_point_shape_index, n_part_3, delete_0=FLAGS.delete_0)
      miou_v2_3, iou_v2_3_value = compute_iou_v2(all_point_predict_part_index_3, all_point_gt_part_index_3, n_part_3, delete_0=FLAGS.delete_0)
      miou_v3_3, iou_v3_3_value = compute_iou_v3(all_point_predict_part_index_3, all_point_gt_part_index_3, all_point_shape_index, n_part_3, delete_0=FLAGS.delete_0)
      miou_v4_3 = compute_iou_v4(all_point_predict_part_index_3, all_point_gt_part_index_3, all_point_shape_index)
      structure_accuracy_3, part_diff_3 = compute_structure_accuracy(all_part_predict_mask_3, all_part_gt_mask_3, delete_0=FLAGS.delete_0)

      result_string = 'Iteration: {}\nmiou v2:\n{:4.1f}\n{:4.1f}\n{:4.1f}\n\nmiouv3:\n{:4.1f}\n{:4.1f}\n{:4.1f}\n\nLevel 1:\n miou v1     : {:5.2f}\n miou v2     : {:5.2f}\n miou v3     : {:5.2f}\n miou v4     : {:5.2f}\n seg accu    : {:6.4f}\n part diff   :  {:6.4f}\n struct accu :  {:6.4f}\n part accu   :  {:6.4f}\n seg loss    :  {:6.4f}\n\nLevel 2:\n miou v1     : {:5.2f}\n miou v2     : {:5.2f}\n miou v3     : {:5.2f}\n miou v4     : {:5.2f}\n seg accu    : {:6.4f}\n part diff   :  {:6.4f}\n struct accu :  {:6.4f}\n part accu   :  {:6.4f}\n seg loss    :  {:6.4f}\n\nLevel 3:\n miou v1     : {:5.2f}\n miou v2     : {:5.2f}\n miou v3     : {:5.2f}\n miou v4     : {:5.2f}\n seg accu    : {:6.4f}\n part diff   :  {:6.4f}\n struct accu :  {:6.4f}\n part accu   :  {:6.4f}\n seg loss    :  {:6.4f}\n'.format(start_iters-1, miou_v2_1*100, miou_v2_2*100, miou_v2_3*100, miou_v3_1*100, miou_v3_2*100, miou_v3_3*100, miou_v1_1*100, miou_v2_1*100, miou_v3_1*100, miou_v4_1*100, avg_seg_accuracy_1, part_diff_1, structure_accuracy_1, avg_part_mask_accuracy_1, avg_seg_loss_1, miou_v1_2*100, miou_v2_2*100, miou_v3_2*100, miou_v4_2*100, avg_seg_accuracy_2, part_diff_2, structure_accuracy_2, avg_part_mask_accuracy_2, avg_seg_loss_2, miou_v1_3*100, miou_v2_3*100, miou_v3_3*100, miou_v4_3*100, avg_seg_accuracy_3, part_diff_3, structure_accuracy_3, avg_part_mask_accuracy_3, avg_seg_loss_3)
      print(result_string); sys.stdout.flush()
      if not os.path.exists(FLAGS.logdir): os.makedirs(FLAGS.logdir)
      with open(os.path.join(FLAGS.logdir, 'test_result_{}.txt'.format(test_iter)), 'w') as f:
        f.write(result_string)


if __name__ == '__main__':
  tf.app.run()
