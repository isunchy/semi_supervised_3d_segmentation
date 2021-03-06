import numpy as np

def compute_iou_v1(point_cube_index, point_part_index, point_shape_index, n_cube, delete_0=False):
  assert(point_cube_index.size == point_part_index.size == point_shape_index.size)
  n_shape = np.max(point_shape_index) + 1
  shape_iou = np.zeros([n_shape])
  valid_shape_num = 0
  for i in range(n_shape):
    shape_point_index = point_shape_index == i
    shape_cube_index = point_cube_index[shape_point_index]
    shape_part_index = point_part_index[shape_point_index]
    if delete_0:
      non_neg_mask = shape_part_index > 0
      shape_cube_index = shape_cube_index[non_neg_mask]
      shape_part_index = shape_part_index[non_neg_mask]
    shape_part_count = 0
    for part_id in np.unique(shape_part_index):
      part_id_point_index_of_part = shape_part_index == part_id
      part_id_point_index_of_cube = shape_cube_index == part_id
      intersection = np.sum(np.logical_and(part_id_point_index_of_part, part_id_point_index_of_cube))
      union = np.sum(np.logical_or(part_id_point_index_of_part, part_id_point_index_of_cube))
      iou = intersection/union
      shape_iou[i] += iou
      shape_part_count += 1
    valid_shape_num += (1 if shape_part_count>0 else 0)
    shape_iou[i] /= (shape_part_count if shape_part_count>0 else 1)
  return np.sum(shape_iou)/valid_shape_num, shape_iou


def compute_iou_v2(point_cube_index, point_part_index, n_cube, delete_0=False):
  assert(point_cube_index.size == point_part_index.size)
  part_intersection = np.zeros([n_cube])
  part_union = np.zeros([n_cube])
  part_flag = np.zeros([n_cube], dtype=int)
  if delete_0:
    non_neg_mask = point_part_index > 0
    point_cube_index = point_cube_index[non_neg_mask]
    point_part_index = point_part_index[non_neg_mask]
  for part_id in range(n_cube):
    part_id_point_index_of_cube = point_cube_index == part_id
    part_id_point_index_of_part = point_part_index == part_id
    intersection = np.sum(np.logical_and(part_id_point_index_of_part, part_id_point_index_of_cube))
    union = np.sum(np.logical_or(part_id_point_index_of_part, part_id_point_index_of_cube))
    if np.sum(part_id_point_index_of_part) > 0: part_flag[part_id] = 1
    part_intersection[part_id] = intersection
    part_union[part_id] = union
  part_iou = part_intersection/(part_union+1e-5)
  if delete_0:
    part_iou = part_iou[1:]
    part_flag = part_flag[1:]
  mean_part_iou = np.sum(part_iou)/np.sum(part_flag)
  return mean_part_iou, part_iou


def compute_iou_v3(point_cube_index, point_part_index, point_shape_index, n_cube, delete_0=False):
  assert(point_cube_index.size == point_part_index.size == point_shape_index.size)
  n_shape = np.max(point_shape_index) + 1
  shape_iou = np.zeros([n_shape])
  valid_shape_num = 0
  for i in range(n_shape):
    shape_point_index = point_shape_index == i
    shape_cube_index = point_cube_index[shape_point_index]
    shape_part_index = point_part_index[shape_point_index]
    if delete_0:
      non_neg_mask = shape_part_index > 0
      shape_cube_index = shape_cube_index[non_neg_mask]
      shape_part_index = shape_part_index[non_neg_mask]
    shape_part_count = 0
    for part_id in np.unique(np.concatenate((shape_part_index, shape_cube_index))):
      part_id_point_index_of_part = shape_part_index == part_id
      part_id_point_index_of_cube = shape_cube_index == part_id
      intersection = np.sum(np.logical_and(part_id_point_index_of_part, part_id_point_index_of_cube))
      union = np.sum(np.logical_or(part_id_point_index_of_part, part_id_point_index_of_cube))
      iou = intersection/union
      shape_iou[i] += iou
      shape_part_count += 1
    valid_shape_num += (1 if shape_part_count>0 else 0)
    shape_iou[i] /= (shape_part_count if shape_part_count>0 else 1)
  return np.sum(shape_iou)/valid_shape_num, shape_iou

def compute_iou_v4(point_cube_index, point_part_index, point_shape_index):
  assert(point_cube_index.size == point_part_index.size == point_shape_index.size)
  n_shape = np.max(point_shape_index) + 1
  n_part = np.max(point_part_index) + 1
  shape_iou = np.zeros([n_shape])
  for i in range(n_shape):
    shape_point_index = point_shape_index == i
    shape_cube_index = point_cube_index[shape_point_index]
    shape_part_index = point_part_index[shape_point_index]
    for part_id in range(n_part):
      part_id_point_index_of_part = shape_part_index == part_id
      part_id_point_index_of_cube = shape_cube_index == part_id
      if np.sum(part_id_point_index_of_part) == 0 and np.sum(part_id_point_index_of_cube) == 0:
        shape_iou[i] += 1.0
      else:
        intersection = np.sum(np.logical_and(part_id_point_index_of_part, part_id_point_index_of_cube))
        union = np.sum(np.logical_or(part_id_point_index_of_part, part_id_point_index_of_cube))
        iou = intersection/union
        shape_iou[i] += iou
    shape_iou[i] /= n_part
  return np.sum(shape_iou)/n_shape


def compute_structure_accuracy(cube_pred_mask, cube_gt_mask, delete_0=False):
  if delete_0:
    # for partnet, 0 for no label ##
    cube_pred_mask = cube_pred_mask[:, 1:]
    cube_gt_mask = cube_gt_mask[:, 1:]
    ###############################
  n_shape, n_cube = cube_gt_mask.shape
  assert(cube_pred_mask.shape == cube_gt_mask.shape)
  matched_predict = cube_pred_mask == cube_gt_mask
  cube_diff = np.sum(matched_predict, axis=1)-n_cube
  structure_accuracy = np.mean((cube_diff==0).astype(np.float32))
  mean_cube_diff = np.mean(np.abs(cube_diff))
  return structure_accuracy, mean_cube_diff
