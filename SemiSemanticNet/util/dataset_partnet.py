import os
import sys
import tensorflow as tf
assert(os.path.isdir('ocnn/tensorflow'))
sys.path.append('ocnn/tensorflow')
from libs import *
import numpy as np

category_label = {
  'Bag': 1,
  'Bed': 2,
  'Bottle': 3,
  'Bowl': 4,
  'Chair': 5,
  'Clock': 6,
  'Dishwasher': 7,
  'Display': 8,
  'Door': 9,
  'Earphone': 10,
  'Faucet': 11,
  'Hat': 12,
  'Keyboard': 13,
  'Knife': 14,
  'Lamp': 15,
  'Laptop': 16,
  'Microwave': 17,
  'Mug': 18,
  'Refrigerator': 19,
  'Scissors': 20,
  'StorageFurniture': 21,
  'Table': 22,
  'TrashCan': 23,
  'Vase': 24,
}

def get_label_mapping(category=1, level=0):
  label_mapping = {
      '1': # Bag
        [
          [
              tf.constant([0,1,2,3], tf.int32), # 4
              tf.constant([0,1,2,3], tf.int32), # 4
          ],
          [
              tf.constant([0,1,2,3], tf.int32), # 4
              tf.constant([0,1,2,3], tf.int32), # 4
          ]
        ],
      '2': # Bed
        [
          [
              tf.constant([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], tf.int32), # 15
              tf.constant([0,1,2,3,3,4,4,4,5,5, 5, 6, 7, 8, 9], tf.int32), # 10
          ],
          [
              tf.constant([0,1,2,3,4,5,6,7,8,9], tf.int32), # 10
              tf.constant([0,1,1,2,2,2,2,2,3,3], tf.int32), # 4
          ]
        ],
      '3': # Bottle
        [
          [
              tf.constant([0,1,2,3,4,5,6,7,8], tf.int32), # 9
              tf.constant([0,1,0,2,3,0,0,4,5], tf.int32), # 6
          ],
          [
              tf.constant([0,1,2,3,4,5], tf.int32), # 6
              tf.constant([0,1,2,3,4,5], tf.int32), # 6
          ]
        ],
      '4': # Bowl
        [
          [
              tf.constant([0,1,2,3], tf.int32), # 4
              tf.constant([0,1,2,3], tf.int32), # 4
          ],
          [
              tf.constant([0,1,2,3], tf.int32), # 4
              tf.constant([0,1,2,3], tf.int32), # 4
          ]
        ],
      '5': # Chair
        [
          [
              tf.constant([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38], tf.int32), # 39
              tf.constant([0,1,2,3,3,3,4,5,6,6, 6, 7, 8, 9,10,11,12,13,13,13,14,15,15,16,17,18,19,20,21,22,23,24,25,25,26,26,27,28,29], tf.int32), # 30
          ],
          [
              tf.constant([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29], tf.int32), # 30
              tf.constant([0,1,1,2,2,2,2,3,3,3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 0, 0, 0], tf.int32), # 6
          ]
        ],
      '6': # Clock
        [
          [
              tf.constant([0,1,2,3,4,5,6,7,8,9,10], tf.int32), # 11
              tf.constant([0,1,1,2,2,3,3,4,5,5, 0], tf.int32), # 6
          ],
          [
              tf.constant([0,1,2,3,4,5], tf.int32), # 6
              tf.constant([0,1,2,3,4,5], tf.int32), # 6
          ]
        ],
      '7': # Dishwasher
        [
          [
              tf.constant([0,1,2,3,4,5,6], tf.int32), # 7
              tf.constant([0,1,2,2,3,3,4], tf.int32), # 5
          ],
          [
              tf.constant([0,1,2,3,4], tf.int32), # 5
              tf.constant([0,1,1,2,2], tf.int32), # 3
          ]
        ],
      '8': # Display
        [
          [
              tf.constant([0,1,2,3], tf.int32), # 4
              tf.constant([0,1,2,2], tf.int32), # 3
          ],
          [
              tf.constant([0,1,2], tf.int32), # 3
              tf.constant([0,1,2], tf.int32), # 3
          ]
        ],
      '9': # Door
        [
          [
              tf.constant([0,1,2,3,4], tf.int32), # 5
              tf.constant([0,1,2,2,3], tf.int32), # 4
          ],
          [
              tf.constant([0,1,2,3], tf.int32), # 4
              tf.constant([0,1,2,2], tf.int32), # 3
          ]
        ],
      '10': # Earphone
        [
          [
              tf.constant([0,1,2,3,4,5,6,7,8,9], tf.int32), # 10
              tf.constant([0,1,1,1,2,3,4,4,4,5], tf.int32), # 6
          ],
          [
              tf.constant([0,1,2,3,4,5], tf.int32), # 6
              tf.constant([0,1,2,3,4,5], tf.int32), # 6
          ]
        ],
      '11': # Faucet
        [
          [
              tf.constant([0,1,2,3,4,5,6,7,8,9,10,11], tf.int32), # 12
              tf.constant([0,1,2,3,4,4,4,5,6,7, 7, 7], tf.int32), # 8
          ],
          [
              tf.constant([0,1,2,3,4,5,6,7], tf.int32), # 8
              tf.constant([0,1,2,3,4,5,6,7], tf.int32), # 8
          ]
        ],
      '12': # Hat
        [
          [
              tf.constant([0,1,2,3,4,5], tf.int32), # 6
              tf.constant([0,1,2,3,4,5], tf.int32), # 6
          ],
          [
              tf.constant([0,1,2,3,4,5], tf.int32), # 6
              tf.constant([0,1,2,3,4,5], tf.int32), # 6
          ]
        ],
      '13': # Keyboard
        [
          [
              tf.constant([0,1,2], tf.int32), # 3
              tf.constant([0,1,2], tf.int32), # 3
          ],
          [
              tf.constant([0,1,2], tf.int32), # 3
              tf.constant([0,1,2], tf.int32), # 3
          ]
        ],
      '14': # Knife
        [
          [
              tf.constant([0,1,2,3,4,5,6,7,8,9], tf.int32), # 10
              tf.constant([0,1,1,1,2,3,3,3,4,4], tf.int32), # 5
          ],
          [
              tf.constant([0,1,2,3,4], tf.int32), # 5
              tf.constant([0,1,2,3,4], tf.int32), # 5
          ]
        ],
      '15': # Lamp
        [
          [
              tf.constant([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], tf.int32), # 41
              tf.constant([0,1,2,2,3,4,5,6,7,8, 8, 9,10,11,12,13,14,15,16,16,17,17,17,17,18,19,20,21,21,22,23,24,25,26,26,27,27,27,27,27,27], tf.int32), # 28
          ],
          [
              tf.constant([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27], tf.int32), # 28
              tf.constant([0,1,1,2,3,4,5,6,6,7, 8, 9, 9, 9, 9,10,10,10,11,11,12,13,13,14,15,16,17,17], tf.int32), # 18
          ],
        ],
      '16': # Laptop
        [
          [
              tf.constant([0,1,2], tf.int32), # 3
              tf.constant([0,1,2], tf.int32), # 3
          ],
          [
              tf.constant([0,1,2], tf.int32), # 3
              tf.constant([0,1,2], tf.int32), # 3
          ]
        ],
      '17': # Microwave
        [
          [
              tf.constant([0,1,2,3,4,5], tf.int32), # 6
              tf.constant([0,1,2,2,3,4], tf.int32), # 5
          ],
          [
              tf.constant([0,1,2,3,4], tf.int32), # 5
              tf.constant([0,1,1,1,2], tf.int32), # 3
          ]
        ],
      '18': # Mug
        [
          [
              tf.constant([0,1,2,3], tf.int32), # 4
              tf.constant([0,1,2,3], tf.int32), # 4
          ],
          [
              tf.constant([0,1,2,3], tf.int32), # 4
              tf.constant([0,1,2,3], tf.int32), # 4
          ]
        ],
      '19': # Refrigerator
        [
          [
              tf.constant([0,1,2,3,4,5,6], tf.int32), # 7
              tf.constant([0,1,2,2,3,4,5], tf.int32), # 6
          ],
          [
              tf.constant([0,1,2,3,4,5], tf.int32), # 6
              tf.constant([0,1,1,1,2,2], tf.int32), # 3
          ]
        ],
      '20': # Scissors
        [
          [
              tf.constant([0,1,2], tf.int32), # 3
              tf.constant([0,1,2], tf.int32), # 3
          ],
          [
              tf.constant([0,1,2], tf.int32), # 3
              tf.constant([0,1,2], tf.int32), # 3
          ]
        ],
      '21': # StorageFurniture
        [
          [
              tf.constant([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], tf.int32), # 24
              tf.constant([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,16,17,17,17,18,18,18], tf.int32), # 19
          ],
          [
              tf.constant([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], tf.int32), # 19
              tf.constant([0,1,2,3,3,3,3,3,3,3, 3, 4, 4, 4, 4, 4, 5, 5, 6], tf.int32), # 7
          ],
        ],
      '22': # Table
        [
          [
              tf.constant([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50], tf.int32), # 51
              tf.constant([0,1,2,3,4,5,6,7,8,9,10,11,12,13,13,13,14,15,16,17,18,19,20,21,22,23,24,24,25,26,27,28,29,30,31,31,32,32,32,32,32,33,34,35,36,37,37,38,39,40,41], tf.int32), # 42
          ],
          [
              tf.constant([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41], tf.int32), # 42
              tf.constant([0,1,2,3,3,0,4,4,5,6, 7, 0, 8, 9, 9, 9,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10], tf.int32), # 11
          ]

        ],
      '23': # TrashCan
        [
          [
              tf.constant([0,1,2,3,4,5,6,7,8,9,10], tf.int32), # 11
              tf.constant([0,1,1,1,2,2,2,2,3,4, 4], tf.int32), # 5
          ],
          [
              tf.constant([0,1,2,3,4], tf.int32), # 5
              tf.constant([0,1,2,3,4], tf.int32), # 5
          ],
        ],
      '24': # Vase
        [
          [
              tf.constant([0,1,2,3,4,5], tf.int32), # 6
              tf.constant([0,1,1,2,3,3], tf.int32), # 4
          ],
          [
              tf.constant([0,1,2,3], tf.int32), # 4
              tf.constant([0,1,2,3], tf.int32), # 4
          ],
        ],
  }
  return label_mapping[str(category)][level][1]

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
  def __init__(self, depth, test=False, rotation_angle=10, scale=0.25, jitter=0.25):
    self._depth = depth
    self._test = test
    self._rotation_angle = rotation_angle
    self._scale = scale
    self._jitter = jitter

  def __call__(self, record):
    raw_points_bytes, label_1, label_2, label_3, points_flag, shape_index = self.parse_example(record)
    radius, center = bounding_sphere(raw_points_bytes)
    raw_points_bytes = normalize_points(raw_points_bytes, radius, center)

    # get 3 level labels
    new_points_bytes = self.set_label(raw_points_bytes, label_1, label_2, label_3)

    if self._test:
      octree = points2octree(raw_points_bytes, depth=self._depth, full_depth=2, node_dis=True)
      return octree, new_points_bytes
    else:
      # augment points
      points_bytes = self.augment_points(new_points_bytes) # []
      ghost_points_bytes = self.augment_points(new_points_bytes) # []
      # clip points
      inbox_points_bytes, inbox_mask = self.get_clip_pts(points_bytes) # [], [n_point]
      ghost_inbox_points_bytes, ghost_inbox_mask = self.get_clip_pts(ghost_points_bytes) # [], [n_point]
      # get octree
      inbox_points_bytes_for_oct = self.set_raw_label(inbox_points_bytes)
      ghost_inbox_points_bytes_for_oct = self.set_raw_label(ghost_inbox_points_bytes)
      octree = points2octree(inbox_points_bytes_for_oct, depth=self._depth, full_depth=2, node_dis=True)
      ghost_octree = points2octree(ghost_inbox_points_bytes_for_oct, depth=self._depth, full_depth=2, node_dis=True)
      # get mask bytes
      inbox_mask_bytes = tf.py_func(mask_to_bytes, [inbox_mask], Tout=tf.string)
      ghost_inbox_mask_bytes = tf.py_func(mask_to_bytes, [ghost_inbox_mask], Tout=tf.string)
      return octree, ghost_octree, inbox_points_bytes, ghost_inbox_points_bytes, inbox_mask_bytes, ghost_inbox_mask_bytes, points_flag

  def set_label(self, points_bytes, label_1, label_2, label_3):
    points_pts = points_property(points_bytes, property_name='xyz', channel=3) # [n_point, 3]
    points_normal = points_property(points_bytes, property_name='normal', channel=3) # [n_point, 3]
    new_label = ((label_1*100)+label_2)*100+label_3
    new_points_bytes = points_new(points_pts, points_normal, tf.zeros([0]), tf.cast(new_label, dtype=tf.float32))
    return new_points_bytes

  def set_raw_label(self, points_bytes):
    points_pts = points_property(points_bytes, property_name='xyz', channel=3) # [n_point, 3]
    points_normal = points_property(points_bytes, property_name='normal', channel=3) # [n_point, 3]
    points_label = points_property(points_bytes, property_name='label', channel=1) # [n_point, 1]
    new_label = tf.ones_like(points_label)
    new_points_bytes = points_new(points_pts, points_normal, tf.zeros([0]), tf.cast(new_label, dtype=tf.float32))
    return new_points_bytes

  def augment_points(self, points_bytes):
    rnd = tf.random.uniform(shape=[3], minval=-self._rotation_angle, maxval=self._rotation_angle, dtype=tf.int32)
    angle = tf.cast(rnd, dtype=tf.float32) * 3.14159265 / 180.0
    scale = tf.random.uniform(shape=[3], minval=1-self._scale, maxval=1+self._scale, dtype=tf.float32)
    scale = tf.stack([scale[0]]*3)
    jitter = tf.random.uniform(shape=[3], minval=-self._jitter, maxval=self._jitter, dtype=tf.float32)
    points_pts = points_property(points_bytes, property_name='xyz', channel=3) # [n_point, 3]
    points_normal = points_property(points_bytes, property_name='normal', channel=3) # [n_point, 3]
    points_label = points_property(points_bytes, property_name='label', channel=1) # [n_point, 1]
    transform_matrix, rotation_matrix = tf.py_func(get_transform_matrix, [angle, jitter, scale, True], Tout=[tf.float32, tf.float32]) # [4, 4], [4, 4]
    transformed_points_pts = apply_transform_to_points(transform_matrix, points_pts) # [n_point, 3]
    transformed_points_normal = apply_transform_to_points(rotation_matrix, points_normal) # [n_point, 3]
    points_bytes = points_new(transformed_points_pts, transformed_points_normal, tf.zeros([0]), points_label)
    return points_bytes

  def get_clip_pts(self, points_bytes):
    points_pts = points_property(points_bytes, property_name='xyz', channel=3) # [n_point, 3]
    points_normal = points_property(points_bytes, property_name='normal', channel=3) # [n_point, 3]
    points_label = points_property(points_bytes, property_name='label', channel=1) # [n_point, 1]
    inbox_mask = self.clip_pts(points_pts) # [n_point]
    points_pts = tf.boolean_mask(points_pts, inbox_mask) # [n_inbox_point, 3]
    points_normal = tf.boolean_mask(points_normal, inbox_mask) # [n_inbox_point, 3]
    points_label = tf.boolean_mask(points_label, inbox_mask) # [n_inbox_point, 1]
    points_bytes = points_new(points_pts, points_normal, tf.zeros([0]), points_label)
    return points_bytes, inbox_mask

  def clip_pts(self, pts):
    abs_pts = tf.abs(pts) # [n_point, 3]
    max_value = tf.math.reduce_max(abs_pts, axis=1) # [n_point]
    inbox_mask = tf.cast(max_value <= 1.0, dtype=tf.bool) # [n_point]
    return inbox_mask

  def parse_example(self, record):
    features = {'points_bytes': tf.io.FixedLenFeature([], tf.string),
                'label_1': tf.io.FixedLenFeature([10000], tf.int64),
                'label_2': tf.io.FixedLenFeature([10000], tf.int64),
                'label_3': tf.io.FixedLenFeature([10000], tf.int64),
                'points_flag': tf.io.FixedLenFeature([1], tf.int64),
                'shape_index': tf.io.FixedLenFeature([1], tf.int64)
                 }
    parsed = tf.io.parse_single_example(record, features)
    return [parsed['points_bytes'], parsed['label_1'], parsed['label_2'], parsed['label_3'],
      parsed['points_flag'], parsed['shape_index']]


def points_dataset(record_name, batch_size, depth=6, test=False, rotation_angle=5, scale=0.25, jitter=0.125):
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
    return dataset.map(PointsPreprocessor(depth, test=test, rotation_angle=rotation_angle, scale=scale, jitter=jitter), num_parallel_calls=8).batch(batch_size) \
                  .map(merge_octrees_test if test else merge_octrees_training, num_parallel_calls=8) \
                  .prefetch(8).make_one_shot_iterator().get_next()
