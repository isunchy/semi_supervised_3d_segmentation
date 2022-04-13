import os
import sys
import tensorflow as tf
assert(os.path.isdir('ocnn/tensorflow'))
sys.path.append('ocnn/tensorflow')
from libs import *
import numpy as np
import random
import scipy.interpolate
import scipy.ndimage


category_label = {
  'scene1': 1, # wordnet
  'scene2': 2, # random1
  'scene3': 3, # random2
  'scene4': 4, # random3
}


def get_label_mapping(category=1):
  label_mapping = {
      '1': # wordnet
          np.array([
              [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20], # 21
              [ 0, 1, 2, 3, 3, 3, 3, 3, 4, 1, 4, 4, 3, 3, 5, 4, 5, 6, 6, 5, 4], # 7
          ], dtype=np.int32),
      '2': # random1
          np.array([
              [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20], # 21
              [ 0, 6, 3, 5, 3, 2, 4, 3, 4, 2, 2, 1, 2, 2, 1, 1, 2, 4, 6, 5, 1], # 7
          ], dtype=np.int32),
      '3': # random2
          np.array([
              [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20], # 21
              [ 0, 4, 6, 4, 2, 3, 5, 3, 5, 6, 5, 6, 6, 5, 2, 4, 1, 4, 6, 6, 3], # 7
          ], dtype=np.int32),
      '4': # random3
          np.array([
              [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20], # 21
              [ 0, 4 ,3 ,2 ,5 ,3 ,4 ,5 ,1 ,4 ,5 ,2 ,1 ,1 ,2 ,2 ,2 ,1 ,2 ,5 ,2], # 7
          ], dtype=np.int32)
  }
  return label_mapping[str(category)][1]

def points_label_mapping(points_label, category=1):
  label_mapping = get_label_mapping(category=category)
  return label_mapping[points_label.astype(np.int32)]

def mask_to_bytes(mask):
  return mask.tobytes()

def fill_rotation_matrix(angle):
  cosx = np.cos(angle[0]); sinx = np.sin(angle[0])
  cosy = np.cos(angle[1]); siny = np.sin(angle[1])
  cosz = np.cos(angle[2]); sinz = np.sin(angle[2])
  rotx = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, cosx, sinx, 0.0], [0.0, -sinx, cosx, 0.0], [0.0, 0.0, 0.0, 1.0]])
  roty = np.array([[cosy, 0.0, -siny, 0.0], [0.0, 1.0, 0.0, 0.0], [siny, 0.0, cosy, 0.0], [0.0, 0.0, 0.0, 1.0]])
  rotz = np.array([[cosz, sinz, 0.0, 0.0], [-sinz, cosz, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
  return np.matmul(rotz, np.matmul(rotx, roty)).astype(np.float32)

def fill_translation_matrix(jitter):
  translation_matrix = np.array([[1.0, 0.0, 0.0, jitter[0]], [0.0, 1.0, 0.0, jitter[1]], [0.0, 0.0, 1.0, jitter[2]], [0.0, 0.0, 0.0, 1.0]])
  return translation_matrix.astype(np.float32)

def fill_scale_matrix(scale):
  scale_matrix = np.array([[scale[0], 0.0, 0.0, 0.0], [0.0, scale[1], 0.0, 0.0], [0.0, 0.0, scale[2], 0.0], [0.0, 0.0, 0.0, 1.0]])
  return scale_matrix.astype(np.float32)

def get_transform_matrix(angle, jitter, scale, return_rotation=False):
  rotation_matrix = fill_rotation_matrix(angle)
  translation_matrix = fill_translation_matrix(jitter)
  scale_matrix = fill_scale_matrix(scale)
  transform_matrix = np.matmul(scale_matrix, np.matmul(translation_matrix, rotation_matrix))
  if return_rotation is False:
    return transform_matrix
  else:
    return transform_matrix, rotation_matrix

def apply_transform_to_points(transform_matrix, points):
  points = tf.pad(points, tf.constant([[0, 0], [0, 1]]), constant_values=1.0) # [n_point, 4]
  transformed_points = tf.matmul(transform_matrix, points, transpose_b=True) # [4, n_point]
  return tf.transpose(transformed_points[:3, :])

class PointsPreprocessor:
  def __init__(self, depth, test=False):
    self._depth = depth
    self._test = test
    self._color_trans_ratio = 0.10
    self._color_jitter_std = 0.05
    self._elastic_params = np.array([[0.2, 0.4], [0.8, 1.6]], np.float32)

  def __call__(self, record):
    raw_points_bytes, md5, points_flag, shape_index = self.parse_example(record)
    radius, center = bounding_sphere(raw_points_bytes)
    radius = 5.12
    raw_points_bytes = normalize_points(raw_points_bytes, radius, center)

    if self._test:
      octree = points2octree(raw_points_bytes, depth=self._depth, full_depth=2, node_dis=True)
      return octree, raw_points_bytes
    else:
      # augment points
      points_bytes = self.augment_points(raw_points_bytes) # []
      ghost_points_bytes = self.augment_points(raw_points_bytes) # []

      pts = points_property(points_bytes, property_name='xyz', channel=3) # [n_point, 3]
      point_num = tf.shape(pts)[0]
      perm = tf.random.shuffle(tf.range(point_num))
      sample_index = perm[0:int(tf.cast(point_num, dtype=tf.float32)*0.2)]

      # clip points
      octree_points_bytes, inbox_points_bytes, inbox_mask = self.get_clip_pts(points_bytes, sample_index) # [], [], [10000]
      ghost_octree_points_bytes, ghost_inbox_points_bytes, ghost_inbox_mask = self.get_clip_pts(ghost_points_bytes, sample_index) # [], [], [10000]
      # get octree
      octree = points2octree(octree_points_bytes, depth=self._depth, full_depth=2, node_dis=True)
      ghost_octree = points2octree(ghost_octree_points_bytes, depth=self._depth, full_depth=2, node_dis=True)

      # get mask bytes
      inbox_mask_bytes = tf.py_func(mask_to_bytes, [inbox_mask], Tout=tf.string)
      ghost_inbox_mask_bytes = tf.py_func(mask_to_bytes, [ghost_inbox_mask], Tout=tf.string)
      return octree, ghost_octree, inbox_points_bytes, ghost_inbox_points_bytes, inbox_mask_bytes, ghost_inbox_mask_bytes, points_flag

  def augment_points(self, points_bytes):
    # scene transform
    color = points_property(points_bytes, property_name='feature', channel=3)
    params = [color, self._color_trans_ratio, self._color_jitter_std]
    color = tf.py_func(self.color_distort, params, tf.float32)
    points_bytes = points_set_property(points_bytes, color, property_name='feature')

    xyz = points_property(points_bytes, property_name='xyz', channel=3)
    params = [xyz, self._elastic_params]
    xyz = tf.py_func(self.elastic_distort, params, tf.float32)
    points_bytes = points_set_property(points_bytes, xyz, property_name='xyz')

    rotation_angle = 180
    rnd = tf.random.uniform(shape=[3], minval=-rotation_angle, maxval=rotation_angle, dtype=tf.int32)
    angle = tf.cast(rnd, dtype=tf.float32) * tf.constant([1./64., 1./64., 1.]) * 3.14159265 / 180.0
    scale = tf.random.uniform(shape=[3], minval=0.9, maxval=1.1, dtype=tf.float32)
    scale = tf.stack([scale[0]]*3)
    jitter = tf.random.uniform(shape=[3], minval=-0.2, maxval=0.2, dtype=tf.float32)* tf.constant([1., 1., 0.])
    points_pts = points_property(points_bytes, property_name='xyz', channel=3) # [n_point, 3]
    points_normal = points_property(points_bytes, property_name='normal', channel=3) # [n_point, 3]
    points_label = points_property(points_bytes, property_name='label', channel=1) # [n_point, 1]
    points_feature = points_property(points_bytes, property_name='feature', channel=3) # [n_point, 3]
    transform_matrix, rotation_matrix = tf.py_func(get_transform_matrix, [angle, jitter, scale, True], Tout=[tf.float32, tf.float32]) # [4, 4], [4, 4]
    transformed_points_pts = apply_transform_to_points(transform_matrix, points_pts) # [n_point, 3]
    transformed_points_normal = apply_transform_to_points(rotation_matrix, points_normal) # [n_point, 3]
    points_bytes = points_new(transformed_points_pts, transformed_points_normal, points_feature, points_label)
    return points_bytes

  def color_distort(self, color, trans_range_ratio, jitter_std):
    color = color * 255.0
    color = self._color_autocontrast(color)
    color = self._color_translation(color, trans_range_ratio)
    color = self._color_jiter(color, jitter_std)
    color = color / 255.0
    return color

  def elastic_distort(self, points, distortion_params):
    if distortion_params.shape[0] > 0:
      assert distortion_params.shape[1] == 2
      if random.random() < 0.95:
        for granularity, magnitude in distortion_params:
          points = self._elastic_distort(points, granularity, magnitude)
    return points

  def _color_translation(self, color, trans_range_ratio=0.1):
    assert color.shape[1] >= 3
    if random.random() < 0.95:
      tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * trans_range_ratio
      color[:, :3] = np.clip(tr + color[:, :3], 0, 255)
    return color

  def _color_autocontrast(self, color, rand_blend_factor=True, blend_factor=0.5):
    assert color.shape[1] >= 3
    lo = color[:, :3].min(0, keepdims=True)
    hi = color[:, :3].max(0, keepdims=True)
    assert hi.max() > 1

    scale = 255 / (hi - lo)
    contrast_feats = (color[:, :3] - lo) * scale

    blend_factor = random.random() if rand_blend_factor else blend_factor
    color[:, :3] = (1 - blend_factor) * color + blend_factor * contrast_feats
    return color

  def _color_jiter(self, color, std=0.01):
    if random.random() < 0.95:
      noise = np.random.randn(color.shape[0], 3)
      noise *= std * 255
      color[:, :3] = np.clip(noise + color[:, :3], 0, 255)
    return color

  def _elastic_distort(self, coords, granularity, magnitude):
    blurx = np.ones((3, 1, 1, 1)).astype('float32') / 3
    blury = np.ones((1, 3, 1, 1)).astype('float32') / 3
    blurz = np.ones((1, 1, 3, 1)).astype('float32') / 3
    coords_min = coords.min(0)

    noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
    noise = np.random.randn(*noise_dim, 3).astype(np.float32)

    # Smoothing.
    convolve = scipy.ndimage.filters.convolve
    for _ in range(2):
      noise = convolve(noise, blurx, mode='constant', cval=0)
      noise = convolve(noise, blury, mode='constant', cval=0)
      noise = convolve(noise, blurz, mode='constant', cval=0)

    # Trilinear interpolate noise filters for each spatial dimensions.
    ax = [np.linspace(d_min, d_max, d)
          for d_min, d_max, d in zip(coords_min - granularity,
                                     coords_min + granularity*(noise_dim - 2),
                                     noise_dim)]

    interp = scipy.interpolate.RegularGridInterpolator(
        ax, noise, bounds_error=0, fill_value=0)
    coords += interp(coords) * magnitude
    return coords

  def get_clip_pts(self, points_bytes, sample_index):
    orig_points_pts = points_property(points_bytes, property_name='xyz', channel=3) # [n_point, 3]
    orig_points_normal = points_property(points_bytes, property_name='normal', channel=3) # [n_point, 3]
    orig_points_feature = points_property(points_bytes, property_name='feature', channel=3) # [n_point, 3]
    orig_points_label = points_property(points_bytes, property_name='label', channel=1) # [n_point, 1]

    inbox_mask = self.clip_pts(orig_points_pts) # [n_point]

    octree_points_pts = tf.boolean_mask(orig_points_pts, inbox_mask) # [n_inbox_point, 3]
    octree_points_normal = tf.boolean_mask(orig_points_normal, inbox_mask) # [n_inbox_point, 3]
    octree_points_feature = tf.boolean_mask(orig_points_feature, inbox_mask) # [n_inbox_point, 3]
    octree_points_label = tf.boolean_mask(orig_points_label, inbox_mask) # [n_inbox_point, 1]
    octree_points_bytes = points_new(octree_points_pts, octree_points_normal, octree_points_feature, octree_points_label)

    sampled_mask = tf.gather(inbox_mask, sample_index, axis=0) # [10000]
    points_pts = tf.boolean_mask(tf.gather(orig_points_pts, sample_index, axis=0), sampled_mask) # [n_inbox_point, 3]
    points_normal = tf.boolean_mask(tf.gather(orig_points_normal, sample_index, axis=0), sampled_mask) # [n_inbox_point, 3]
    points_feature = tf.boolean_mask(tf.gather(orig_points_feature, sample_index, axis=0), sampled_mask) # [n_inbox_point, 3]
    points_label = tf.boolean_mask(tf.gather(orig_points_label, sample_index, axis=0), sampled_mask) # [n_inbox_point, 1]
    points_bytes = points_new(points_pts, points_normal, points_feature, points_label)

    return octree_points_bytes, points_bytes, sampled_mask

  def clip_pts(self, pts):
    abs_pts = tf.abs(pts) # [n_point, 3]
    max_value = tf.math.reduce_max(abs_pts, axis=1) # [n_point]
    inbox_mask = tf.cast(max_value <= 1.0, dtype=tf.bool) # [n_point]
    return inbox_mask

  def parse_example(self, record):
    features = {'points_bytes': tf.io.FixedLenFeature([], tf.string),
                'md5': tf.io.FixedLenFeature([], tf.string),
                'points_flag': tf.io.FixedLenFeature([1], tf.int64),
                'shape_index': tf.io.FixedLenFeature([1], tf.int64)
                 }
    parsed = tf.io.parse_single_example(record, features)
    return [parsed['points_bytes'], parsed['md5'], parsed['points_flag'],
      parsed['shape_index']]


def points_dataset(record_name, batch_size, depth=6, test=False):
  def merge_octrees_training(octrees, ghost_octrees, inbox_points_bytes, ghost_inbox_points_bytes,
      inbox_mask_bytes, ghost_inbox_mask_bytes, points_flag):
    octree = octree_batch(octrees)
    ghost_octree = octree_batch(ghost_octrees)
    return [octree, ghost_octree, inbox_points_bytes, ghost_inbox_points_bytes,
      inbox_mask_bytes, ghost_inbox_mask_bytes, points_flag]
  def merge_octrees_test(octrees, points_bytes):
    octree = octree_batch(octrees)
    return octree, points_bytes
  with tf.name_scope('points_dataset'):
    dataset = tf.data.TFRecordDataset([record_name]).repeat()
    if test is False:
      dataset = dataset.shuffle(1000)
    return dataset.map(PointsPreprocessor(depth, test=test), num_parallel_calls=8).batch(batch_size) \
                  .map(merge_octrees_test if test else merge_octrees_training, num_parallel_calls=8) \
                  .prefetch(8).make_one_shot_iterator().get_next()
