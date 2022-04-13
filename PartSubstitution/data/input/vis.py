import os
import numpy as np


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
 

def save_ply_data_numpy(filename, array):
  f = open(filename, 'w')
  f.write('ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty float opacity\nelement face 0\nproperty list uchar int vertex_indices\nend_header\n'.format(array.shape[0]))
  for i in range(array.shape[0]):
    for j in range(6):
      f.write("{:f} ".format(array[i][j]))
    for j in range(3):
      f.write("{:d} ".format(int(array[i][j+6])))
    f.write('1\n')
  f.close()


def export_ply(filename, n_color, points, normals, labels):
  np.random.seed(1)
  colormap = np.round(255* np.random.rand(n_color,3))
  verts_color = np.round(colormap[labels])
  output_data = np.concatenate((points, normals, verts_color), 1)
  save_ply_data_numpy(filename, output_data)


if __name__ == '__main__':
  filelist = [file for file in os.listdir('.') if file.endswith('.points')]
  print(filelist)
  for file in filelist:
    data_list = decode_bytes_v2(file)
    points, normals, labels = data_list[0], data_list[1], data_list[-1].reshape(-1).astype('int32')
    export_ply(file.replace('.points', '.ply'), int(np.max(labels))+1, points, normals, labels)
