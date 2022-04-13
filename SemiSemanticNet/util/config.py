import os
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf_flags = tf.app.flags

tf_flags.DEFINE_string('logdir', 'log/test', 'Directory where to write event logs.')
tf_flags.DEFINE_string('train_data', '', 'Training data.')
tf_flags.DEFINE_string('test_data', '', 'Test data.')
tf_flags.DEFINE_string('test_data_visual', '', 'Testing data for visualization.')
tf_flags.DEFINE_integer('train_batch_size', 16, 'Batch size for the training.')
tf_flags.DEFINE_integer('test_batch_size', 1, 'Batch size for the testing.')
tf_flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
tf_flags.DEFINE_string('optimizer', 'sgd', 'Optimizer (adam/sgd).')
tf_flags.DEFINE_string('decay_policy', 'step', 'Learning rate decay policy (step/poly/constant).')
tf_flags.DEFINE_float('weight_decay', 0.0001, 'Weight decay.')
tf_flags.DEFINE_integer('max_iter', 80000, 'Maximum training iterations.')
tf_flags.DEFINE_integer('test_every_iter', 5000, 'Test model every n training steps.')
tf_flags.DEFINE_integer('test_iter', 100, '#shapes in test data.')
tf_flags.DEFINE_integer('test_iter_visual', 20, 'Test steps in testing phase for visualization.')
tf_flags.DEFINE_boolean('test_visual', False, """Test with visualization.""")
tf_flags.DEFINE_integer('disp_every_n_steps', 80000, 'Visualization every n training steps.')
tf_flags.DEFINE_string('cache_folder', 'test', 'Directory where to dump immediate data.')
tf_flags.DEFINE_string('ckpt', '', 'Restore weights from checkpoint file.')
tf_flags.DEFINE_string('gpu', '0', 'The gpu index.')
tf_flags.DEFINE_string('phase', 'train', 'train/test}.')
tf_flags.DEFINE_integer('n_part', 20, 'Number of semantic part.')
tf_flags.DEFINE_integer('n_part_1', 6, 'Number of semantic part in level one.')
tf_flags.DEFINE_integer('n_part_2', 30, 'Number of semantic part in level two.')
tf_flags.DEFINE_integer('n_part_3', 39, 'Number of semantic part in level three.')
tf_flags.DEFINE_boolean('delete_0', False, """Whether consider label 0 in metric computation.""")
tf_flags.DEFINE_float('seg_loss_weight', 1.0, 'Weight of segmentation loss.')
tf_flags.DEFINE_float('point_consistency_weight', 0.01, 'Weight of point consistency loss.')
tf_flags.DEFINE_float('part_consistency_weight', 0.01, 'Weight of part consistency loss.')
tf_flags.DEFINE_float('hierarchy_point_consistency_weight', 0.01, 'Weight of hierarchy point consistency loss.')
tf_flags.DEFINE_float('level_0_weight', 1.0, 'Weight of level 0 loss.')
tf_flags.DEFINE_float('level_1_weight', 1.0, 'Weight of level 1 loss.')
tf_flags.DEFINE_float('level_2_weight', 1.0, 'Weight of level 2 loss.')
tf_flags.DEFINE_float('level_3_weight', 1.0, 'Weight of level 3 loss.')
tf_flags.DEFINE_float('level_01_weight', 1.0, 'Weight of level 01 loss.')
tf_flags.DEFINE_float('level_32_weight', 1.0, 'Weight of level 32 loss.')
tf_flags.DEFINE_float('level_21_weight', 1.0, 'Weight of level 21 loss.')
tf_flags.DEFINE_integer('test_shape_average_point_number', 10000, 'Mean point number of each shape in test phase. Must be greater than real point number.')
tf_flags.DEFINE_integer('depth', 6, 'The depth of octree.')
tf_flags.DEFINE_integer('rotation_angle', 10, 'Rotation angle of augmentation.')
tf_flags.DEFINE_float('scale', 0.25, 'Scale of augmentation.')
tf_flags.DEFINE_float('jitter', 0.25, 'Jitter of augmentation.')
tf_flags.DEFINE_boolean('use_kl', True, 'Use kl divergency in hierarchy loss.')
tf_flags.DEFINE_string('category', 'chair', 'Category.')
tf_flags.DEFINE_boolean('nempty', False, """Use nempty octree node in conv.""")


FLAGS = tf_flags.FLAGS

os.environ["OCTREE_KEY"] = "64"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

max_iter = FLAGS.max_iter
test_iter = FLAGS.test_iter
test_iter_visual = FLAGS.test_iter_visual
n_part = FLAGS.n_part
n_part_1 = FLAGS.n_part_1
n_part_2 = FLAGS.n_part_2
n_part_3 = FLAGS.n_part_3
n_test_point = FLAGS.test_shape_average_point_number
