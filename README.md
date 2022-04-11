# Semi-Supervised 3D Shape Segmentation with Multilevel Consistency and Part Substitution

<img src="consistency_graphical_abstract.png" alt="consistency_graphical_abstract" width=800px; height=331px;/>


## Introduction

This work is based on our CVM 2022 paper. We proposed a semi-supervised learning method for 3D shape semantic segmentation. You can check our [project webpage](https://isunchy.github.io/projects/semi_supervised_3d_segmentation.html) for a quick overview.

The lack of fine-grained 3D shape segmentation data is the main obstacle to developing learning-based 3D segmentation techniques. We propose an effective semi-supervised method for learning 3D segmentations from a few labeled 3D shapes and a large amount of unlabeled 3D data. For the unlabeled data, we present a novel multilevel consistency loss to enforce consistency of network predictions between perturbed copies of a 3D shape at multiple levels: point-level, part-level, and hierarchical level. For the labeled data, we develop a simple yet effective part substitution scheme to augment the labeled 3D shapes with more structural variations to enhance training. Our method has been extensively validated on the task of 3D object semantic segmentation on PartNet and ShapeNetPart, and indoor scene semantic segmentation on ScanNet. It exhibits superior performance to existing semi-supervised and unsupervised pre-training 3D approaches.

In this repository, we release the code and data for training the semi-supervised networks for 3d shape semantic segmentation.

## Citation

If you use our code for research, please cite our paper:
```
@article{sun2022semisupervised,
  title     = {Semi-Supervised 3D Shape Segmentation with Multilevel Consistency and Part Substitution},
  author    = {Sun, Chunyu and Yang, Yiqi and Guo, Haoxiang and Wang, pengshuai and Tong, Xin and Liu, Yang and Shum Heung-Yeung},
  journal   = {Computational Visual Media},
  year      = {2022}
}
```

<!-- ## Setup

Pre-prequisites

        Python == 3.6
        TensorFlow == 1.12
        numpy-quaternion

Compile customized TensorFlow operators

        $ cd cext
        $ mkdir build
        $ cd build
        $ cmake ..
        $ make

## Experiments


### Data Preparation

Now we provide the Google drive link for downloading the training datasets:

>[Training data](https://drive.google.com/drive/folders/1Uh_-CrOyUVpB5mWkSOY-L-b2kpa9LuTC?usp=sharing)

### Training

To start the training, run

        $ python training_script.py --category class_name

### Test

To test a trained model, run

        $ python iterative_training.py --test_data test_tfrecords --test_iter number_of_shapes --ckpt /path/to/snapshots --cache_folder /path/to/save/test_results --test

Now we provide the trained weights and the final results used in our paper:

>[Weights](https://drive.google.com/drive/folders/1ipixLDU4LejE57R8dnLJFTvGvnkilfv_?usp=sharing)

>[Results](https://drive.google.com/drive/folders/1e_qdJeFtNoPy8jtBKtpMoln-Cfrn8Jya?usp=sharing)
 

## License

MIT Licence -->

## Contact

Please contact us (Chunyu Sun sunchyqd@gmail.com, Yang Liu yangliu@microsoft.com) if you have any problem about our implementation.

