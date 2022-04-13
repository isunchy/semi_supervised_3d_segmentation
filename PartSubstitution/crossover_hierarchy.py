import os
import sys
import numpy as np
import trimesh
import argparse
import random
import copy
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--indir', type=str, default='data/input', help='input data directory')
parser.add_argument('--outdir', type=str, default='data/output', help='output directory')
parser.add_argument('-m', type=int, default=5, help='number of generation')
parser.add_argument('-p', type=float, default=0.5, help='probability of replacement')
parser.add_argument('--cat', type=str, default='chair', help='category')
parser.add_argument('--filelist', type=str, default='input_filelist.txt', help='data file list')
args = parser.parse_args()

EPS = 1e-6

def decode_bytes_v2(path):
  f = open(path, "rb")
  magic_str_ = f.read(16)
  points_num = np.frombuffer(f.read(4), dtype=np.int32)[0]
  int_flags_ = np.frombuffer(f.read(4), dtype=np.int32)[0]
  content_flags_ = [int_flags_%2, (int_flags_>>1)%2, (int_flags_>>2)%2, (int_flags_>>3)%2]
  channels_ = np.frombuffer(f.read(4 * 8), dtype=np.int32)
  ptr_dis_ = np.frombuffer(f.read(4 * 8), dtype=np.int32)
  points_data_list = []
  for c_index in range(4):
      buffer = f.read(int(ptr_dis_[c_index+1] - ptr_dis_[c_index]))
      points_data = np.frombuffer(buffer, dtype=np.float32)
      points_data = points_data[:points_num * channels_[c_index]]
      points_data = np.reshape(points_data, (points_num, channels_[c_index]))
      points_data_list.append(points_data)
  return points_data_list


def points_label_mapping_partnet(cat):
  label_mapping = {
  'chair' :
    np.array([
      [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38], # 39
      [0,1,2,3,3,3,4,5,6,6, 6, 7, 8, 9,10,11,12,13,13,13,14,15,15,16,17,18,19,20,21,22,23,24,25,25,26,26,27,28,29], # 30
      [0,1,1,2,2,2,2,2,2,2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 0, 0, 0], # 6
    ], dtype='int32'),
  'table' :
    np.array([
	    [0,1,2,3,4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50], # 51
	    [0,1,2,3,4, 5,6,7,8,9,10,11,12,13,13,13,14,15,16,17,18,19,20,21,22,23,24,24,25,26,27,28,29,30,31,31,32,32,32,32,32,33,34,35,36,37,37,38,39,40,41], # 42
	    [0,1,2,3,3, 0,4,4,5,6, 7, 0, 8, 9, 9, 9, 9, 9,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10], # 11
	  ], dtype='int32'),
  'lamp' :
    np.array([
	    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40], # 41
	    [0,1,2,2,3,4,5,6,7,8, 8, 9,10,11,12,13,14,15,16,16,17,17,17,17,18,19,20,21,21,22,23,24,25,26,26,27,27,27,27,27,27], # 28
	    [0,1,1,1,2,3,4,5,6,6, 6, 7, 8, 9, 9, 9, 9,10,10,10,10,10,10,10,11,11,12,13,13,13,14,15,16,17,17,17,17,17,17,17,17], # 18
	  ], dtype='int32'),
  'storage' :
    np.array([
		  [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], # 24
		  [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,16,17,17,17,18,18,18], # 19
		  [0,1,2,3,3,3,3,3,3,3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6], # 7
		], dtype='int32'),
  'bed':
    np.array([
      [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], # 15
      [0,1,2,3,3,4,4,4,5,5, 5, 6, 7, 8, 9], # 10
      [0,1,1,2,2,2,2,2,2,2, 2, 2, 2, 3, 3], # 4
    ], dtype=np.int32),
  'bottle':
    np.array([
      [0,1,2,3,4,5,6,7,8], # 9
      [0,1,0,2,3,0,0,4,5], # 6
      [0,1,0,2,3,0,0,4,5], # 6
    ], dtype=np.int32),
  'clock':
    np.array([
      [0,1,2,3,4,5,6,7,8,9,10], # 11
      [0,1,1,2,2,3,3,4,5,5, 0], # 6
      [0,1,1,2,2,3,3,4,5,5, 0], # 6
    ], dtype=np.int32),
  'dishwasher':
    np.array([
      [0,1,2,3,4,5,6], # 7
      [0,1,2,2,3,3,4], # 5
      [0,1,1,1,2,2,2], # 3
    ], dtype=np.int32),
  'display':
    np.array([
      [0,1,2,3], # 4
      [0,1,2,2], # 3
      [0,1,2,2], # 3
    ], dtype=np.int32),
  'door':
    np.array([
      [0,1,2,3,4], # 5
      [0,1,2,2,3], # 4
      [0,1,2,2,2], # 3
    ], dtype=np.int32),
  'earphone':
    np.array([
      [0,1,2,3,4,5,6,7,8,9], # 10
      [0,1,1,1,2,3,4,4,4,5], # 6
      [0,1,1,1,2,3,4,4,4,5], # 6
    ], dtype=np.int32),
  'faucet':
    np.array([
      [0,1,2,3,4,5,6,7,8,9,10,11], # 12
      [0,1,2,3,4,4,4,5,6,7, 7, 7], # 8
      [0,1,2,3,4,4,4,5,6,7, 7, 7], # 8
    ], dtype=np.int32),
  'knife':
    np.array([
      [0,1,2,3,4,5,6,7,8,9], # 10
      [0,1,1,1,2,3,3,3,4,4], # 5
      [0,1,1,1,2,3,3,3,4,4], # 5
    ], dtype=np.int32),
  'microwave':
    np.array([
      [0,1,2,3,4,5], # 6
      [0,1,2,2,3,4], # 5
      [0,1,1,1,1,2], # 3
    ], dtype=np.int32),
  'refrigerator':
    np.array([
      [0,1,2,3,4,5,6], # 7
      [0,1,2,2,3,4,5], # 6
      [0,1,1,1,1,2,2], # 3
    ], dtype=np.int32),
  'trashcan':
    np.array([
      [0,1,2,3,4,5,6,7,8,9,10], # 11
      [0,1,1,1,2,2,2,2,3,4, 4], # 5
      [0,1,1,1,2,2,2,2,3,4, 4], # 5
    ], dtype=np.int32),
  'vase':
    np.array([
      [0,1,2,3,4,5], # 6
      [0,1,1,2,3,3], # 4
      [0,1,1,2,3,3], # 4
    ], dtype=np.int32),
  'bag':
    np.array([
      [0,1,2,3], # 4
      [0,1,2,3], # 4
      [0,1,2,3], # 4
    ], dtype=np.int32),
  'bowl':
    np.array([
      [0,1,2,3], # 4
      [0,1,2,3], # 4
      [0,1,2,3], # 4
    ], dtype=np.int32),
  'hat':
    np.array([
      [0,1,2,3,4,5], # 6
      [0,1,2,3,4,5], # 6
      [0,1,2,3,4,5], # 6
    ], dtype=np.int32),
  'keyboard':
    np.array([
      [0,1,2], # 3
      [0,1,2], # 3
      [0,1,2], # 3
    ], dtype=np.int32),
  'laptop':
    np.array([
      [0,1,2], # 3
      [0,1,2], # 3
      [0,1,2], # 3
    ], dtype=np.int32),
  'mug':
    np.array([
      [0,1,2,3], # 4
      [0,1,2,3], # 4
      [0,1,2,3], # 4
    ], dtype=np.int32),
  'scissors':
    np.array([
      [0,1,2], # 3
      [0,1,2], # 3
      [0,1,2], # 3
    ], dtype=np.int32),
  }
  return label_mapping[cat]


def get_transform_scale_translation(source_list, target_list, source_ref_list, bb_min, bb_max):
  source_verts = np.concatenate(source_list)
  target_verts = np.concatenate(target_list)
  #translate source verts so that its bbs and centers are the same with target verts 
  # axis aligned bb version
  source_verts_min = np.min(source_verts, 0)
  source_verts_max = np.max(source_verts, 0)
  target_verts_min = np.min(target_verts, 0)
  target_verts_max = np.max(target_verts, 0)
  source_extents = source_verts_max - source_verts_min
  target_extents = target_verts_max - target_verts_min
  source_bbcenter = (source_verts_min + source_verts_max)/2
  target_bbcenter = (target_verts_min + target_verts_max)/2
  scale = target_extents / (source_extents + EPS)
  assert(scale.shape[0] == 3)

  source_ref_verts = np.concatenate(source_ref_list)
  source_ref_verts_min = np.min(source_ref_verts, 0)
  source_ref_verts_max = np.max(source_ref_verts, 0)
  source_ref_verts_extents = source_ref_verts_max - source_ref_verts_min
  source_ref_verts_transform = (source_ref_verts - source_bbcenter) * scale + target_bbcenter
  tmp_min = np.min(source_ref_verts_transform, 0) + EPS
  tmp_max = np.max(source_ref_verts_transform, 0) - EPS
  for i in range(3):
    if tmp_min[i] < bb_min[i] or tmp_max[i] > bb_max[i]:
      print('scale redefined: ', i)
      print('bbmin: ', bb_min, " bbmax: ", bb_max)
      print("tmmin: ", tmp_min, " tmmax: ", tmp_max)
      # scale[i] = 1.0
      scale[i] = (bb_max[i] - bb_min[i]) / (source_ref_verts_extents[i] + EPS)
      #translation is unchangable cause we must map common part to its target
  return scale, target_bbcenter - source_bbcenter * scale


def save_ply_data_numpy(filename, array):
  f = open(filename, 'w')
  f.write('ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty float opacity\nelement face 0\nproperty list uchar int vertex_indices\nend_header\n'.format(array.shape[0]))
  for i in range(array.shape[0]):
    for j in range(6):
      f.write("{:f} ".format(array[i][j]))
    for j in range(3):
      f.write("{:d} ".format(int(array[i][j+6])))
    f.write('{:f}\n'.format(array[i][9]))
  f.close()


#functions for pts
class PointsMask():
  def __init__(self, vertices, vertices_normal, vertices_mask, meshid = -1):
    #mask defined 
    assert(vertices.shape[1] == 3)
    assert(vertices_normal.shape[1] == 3)
    self.vertices = vertices
    self.vertices_normal = vertices_normal
    self.vertices_mask=vertices_mask
    self.meshid = -1

  def sample_pts(self, n_sample = 10000):
    if (self.vertices.shape[0] > n_sample):
      sample_indices = np.random.choice(self.vertices.shape[0], n_sample, replace=False)
      self.vertices = self.vertices[sample_indices]
      self.vertices_normal = self.vertices_normal[sample_indices]
      self.vertices_mask = self.vertices_mask[sample_indices]
    else:
      # assert(2 * self.vertices.shape[0] > n_sample)
      if 2 * self.vertices.shape[0] > n_sample:
        sample_indices = np.random.choice(self.vertices.shape[0], n_sample - self.vertices.shape[0], replace=False)
        self.vertices = np.concatenate((self.vertices, self.vertices[sample_indices]))
        self.vertices_normal = np.concatenate((self.vertices_normal, self.vertices_normal[sample_indices]))
        self.vertices_mask = np.concatenate((self.vertices_mask, self.vertices_mask[sample_indices]))
      else:
        sample_indices = np.random.choice(self.vertices.shape[0], n_sample, replace=True)
        self.vertices = self.vertices[sample_indices]
        self.vertices_normal = self.vertices_normal[sample_indices]
        self.vertices_mask = self.vertices_mask[sample_indices]

  def export_pts(self, filename):
    np.savetxt(filename, np.concatenate((self.vertices, self.vertices_normal,self.vertices_mask.astype('float64').reshape(-1,1)),1), fmt='%1.4f')
  
  def export_ply(self, filename, n_color):
    np.random.seed(1)
    colormap = np.round(255* np.random.rand(n_color,3))
    verts_color = np.round(colormap[self.vertices_mask])
    output_data = np.concatenate((self.vertices, self.vertices_normal, verts_color, np.ones([self.vertices.shape[0], 1])), 1)
    save_ply_data_numpy(filename, output_data)


def get_pid2mid2pts_dict(pts, pts_mask, cat, level=0, meshid = -1):
  d = {}
  mask_set = set(pts_mask.flatten())
  pid2mid = {} #part id to mask id, one part may corresponds to multiple masks
  mask2part = points_label_mapping_partnet(cat)
  for mid in mask_set:
    pid = mask2part[level][mid]
    if pid in pid2mid:
      pid2mid[pid].append(mid)
    else:
      pid2mid[pid] = [mid]

  for pid in pid2mid:
    mid2pts = {}
    for mid in pid2mid[pid]:
      flag_mask = (pts_mask==mid)
      part_pts = pts[flag_mask]
      part_mask = mid * np.ones(part_pts.shape[0], dtype='int32')
      mid2pts[mid] = PointsMask(part_pts[:,:3], part_pts[:,3:], part_mask)
      if meshid != -1:
        mid2pts[mid].meshid = meshid
    d[pid] = mid2pts
  return d


def get_pointsmask_from_pid2mid2pts_dict(pid2mid2pts_dict):
  verts = []
  verts_normal = []
  masks = []
  for pid in pid2mid2pts_dict:
    for mid in pid2mid2pts_dict[pid]:
      verts.append(pid2mid2pts_dict[pid][mid].vertices)
      verts_normal.append(pid2mid2pts_dict[pid][mid].vertices_normal)
      masks.append(pid2mid2pts_dict[pid][mid].vertices_mask)
  verts = np.concatenate(verts)
  verts_normal = np.concatenate(verts_normal)
  masks = np.concatenate(masks)
  # return trimesh.Trimesh(vertices=verts, faces=faces)
  return PointsMask(vertices = verts, vertices_normal = verts_normal, vertices_mask = masks)


def main_pts_hierarchy():
  if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

  mask2part = points_label_mapping_partnet(args.cat)
  n_color = np.max(mask2part[0]) + 1
  N_sample_output = 10000
  all_pid2mid2mesh_list_level=[]
  for i in range(mask2part.shape[0]):
    all_pid2mid2mesh_list_level.append([])

  structure_set_before = set()
  structure_set_after = set()

  bb_min_list = []
  bb_max_list = []

  f = open(args.filelist, 'r')
  fns = f.readlines()
  f.close()
  N_total = len(fns)
  for id in range(len(fns)):
    data_list = decode_bytes_v2(os.path.join(args.indir, fns[id].strip()))
    cur_mesh = np.concatenate((data_list[0], data_list[1]), 1)
    cur_mask = data_list[-1].reshape(-1).astype('int32')
    bb_min_list.append(np.min(cur_mesh[:,:3], 0))
    bb_max_list.append(np.max(cur_mesh[:,:3], 0))
    for i in range(mask2part.shape[0]):
      all_pid2mid2mesh_list_level[i].append(get_pid2mid2pts_dict(cur_mesh, cur_mask,args.cat, i, id)) #pid=mid
    structure_set_before.add(frozenset(all_pid2mid2mesh_list_level[0][id].keys()))
    print ('{} set: '.format(id), frozenset(all_pid2mid2mesh_list_level[0][id].keys()));sys.stdout.flush()

  print("constructing parts 2 id finished")
  print("n structure before: ", len(structure_set_before))
  count = 0
  for item in structure_set_before:
    print("count {} :".format(count), item)
    count = count + 1

  for gid in tqdm(range(args.m)):
    tid = round(random.uniform(0.0, 1.0) * (N_total - 1))
    print("-----generation No.{}, template id : {}".format(gid, tid));sys.stdout.flush()
    t_pid2mid2mesh_level0 = all_pid2mid2mesh_list_level[0][tid]
    t_pid2midkeys_level = []
    for i in range(mask2part.shape[0]):
      t_pid2midkeys_level.append({})
      for pid in all_pid2mid2mesh_list_level[i][tid]:
        t_pid2midkeys_level[i][pid] = set(all_pid2mid2mesh_list_level[i][tid][pid].keys())

    other_pid2meshid_level = []
    for i in range(mask2part.shape[0]):
      other_pid2meshid_level.append([])
    for i in range(mask2part.shape[0]):
      #four layers
      N_part = int(np.max(mask2part[i])) + 1
      for j in range(N_part):
        other_pid2meshid_level[i].append([])
      for meshid in range(N_total):
        if meshid == tid:
          continue
        for pid in all_pid2mid2mesh_list_level[i][meshid]:
          if pid == 0:
            continue

          if pid in t_pid2midkeys_level[i]:
            cur_pid2midkeys = set(all_pid2mid2mesh_list_level[i][meshid][pid].keys())
            other_pid2meshid_level[i][pid].append(meshid)

    new_pid2mid2mesh_level0 = copy.deepcopy(t_pid2mid2mesh_level0)
    part_process_flag = {}
    for mid in new_pid2mid2mesh_level0:
      part_process_flag[mid] = False

    n_layer = mask2part.shape[0]
    for i in range(0, n_layer):
      cur_layer = mask2part.shape[0] - 1 - i
      print ("computing for level {}:".format(cur_layer))
      for pid in t_pid2midkeys_level[cur_layer]:
        if pid == 0:
          continue

        if (len(other_pid2meshid_level[cur_layer][pid]) == 0):
          print("pid {} does not exist on other shapes".format(pid))
          continue
        sample_p = random.uniform(0.0, 1.0)
        if sample_p > args.p:
          continue
        if part_process_flag[list(all_pid2mid2mesh_list_level[cur_layer][tid][pid].keys())[0]] == True:
          continue

        for mid in all_pid2mid2mesh_list_level[cur_layer][tid][pid]:
          part_process_flag[mid] = True
        rand_id = random.randrange(len(other_pid2meshid_level[cur_layer][pid]))
        rand_meshid = other_pid2meshid_level[cur_layer][pid][rand_id]
        print ("random mesh id for part {}: {}".format(pid, rand_meshid))

        common_mid = set(all_pid2mid2mesh_list_level[cur_layer][rand_meshid][pid].keys()).intersection(t_pid2midkeys_level[cur_layer][pid])

        if len(common_mid) == 0:
          source_pts_list = []
          target_pts_list = []
          source_pts_ref_list = []
          for mid in all_pid2mid2mesh_list_level[cur_layer][rand_meshid][pid]:
            source_pts_ref_list.append(all_pid2mid2mesh_list_level[cur_layer][rand_meshid][pid][mid].vertices)
            source_pts_list.append(all_pid2mid2mesh_list_level[cur_layer][rand_meshid][pid][mid].vertices)

          for mid in all_pid2mid2mesh_list_level[cur_layer][tid][pid]:
            target_pts_list.append(all_pid2mid2mesh_list_level[cur_layer][tid][pid][mid].vertices)

          scale, translation = get_transform_scale_translation(source_pts_list, target_pts_list, source_pts_ref_list, bb_min_list[tid], bb_max_list[tid])
          #replacement
          for mid in all_pid2mid2mesh_list_level[cur_layer][tid][pid]:
            del new_pid2mid2mesh_level0[mid]

          for mid in all_pid2mid2mesh_list_level[cur_layer][rand_meshid][pid]:
            new_smallmesh = copy.deepcopy(all_pid2mid2mesh_list_level[cur_layer][rand_meshid][pid][mid])
            new_smallmesh.vertices = new_smallmesh.vertices * scale + translation
            if mid in new_pid2mid2mesh_level0:
              new_pid2mid2mesh_level0[mid][mid] = new_smallmesh
            else:
              new_pid2mid2mesh_level0[mid] = {}
              new_pid2mid2mesh_level0[mid][mid] = new_smallmesh
        else:
          source_pts_list = []
          target_pts_list = []
          source_pts_ref_list = []
          for mid in all_pid2mid2mesh_list_level[cur_layer][rand_meshid][pid]:
            source_pts_ref_list.append(all_pid2mid2mesh_list_level[cur_layer][rand_meshid][pid][mid].vertices)

          for mid in common_mid:
            source_pts_list.append(all_pid2mid2mesh_list_level[cur_layer][rand_meshid][pid][mid].vertices)
            target_pts_list.append(t_pid2mid2mesh_level0[mid][mid].vertices)
          scale, translation = get_transform_scale_translation(source_pts_list, target_pts_list, source_pts_ref_list, bb_min_list[tid], bb_max_list[tid])
          #replacement
          for mid in all_pid2mid2mesh_list_level[cur_layer][tid][pid]:
            del new_pid2mid2mesh_level0[mid]

          for mid in all_pid2mid2mesh_list_level[cur_layer][rand_meshid][pid]:
            new_smallmesh = copy.deepcopy(all_pid2mid2mesh_list_level[cur_layer][rand_meshid][pid][mid])
            new_smallmesh.vertices = new_smallmesh.vertices * scale + translation
            if mid in new_pid2mid2mesh_level0:
              new_pid2mid2mesh_level0[mid][mid] = new_smallmesh
            else:
              new_pid2mid2mesh_level0[mid] = {}
              new_pid2mid2mesh_level0[mid][mid] = new_smallmesh

    structure_set_after.add(frozenset(new_pid2mid2mesh_level0.keys()))
    new_mesh = get_pointsmask_from_pid2mid2pts_dict(new_pid2mid2mesh_level0)
    new_mesh.sample_pts(N_sample_output)
    new_mesh.export_pts(os.path.join(args.outdir, '{}.pts'.format(gid)))
    if gid < 10:
      new_mesh.export_ply(os.path.join(args.outdir, '{}.ply'.format(gid)), n_color)

  print("probability: ", args.p)
  print("structure before and after crossover: {} -- {}".format(len(structure_set_before), len(structure_set_after)))


if __name__ == '__main__':
  main_pts_hierarchy()