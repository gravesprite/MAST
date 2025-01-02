# MAST
An efficient Point Cloud query processing framework

## Requirement
``` shell
Cuda driver >= 11.6
docker
OpenPCDet: https://github.com/open-mmlab/OpenPCDet
```

## Installing step by step
``` shell

# Git clone the OpenPCDet repo
git clone https://github.com/open-mmlab/OpenPCDet.git

# Prepare the docker container of the OpenPCDet repo
docker pull XXX/pcdet:0.6.0

# Start the docker, [your path] denotes the directory where you clone the OpenPCDet
docker run -it --gpus all -v [your path]:/root/mast -it XXX/pcdet:0.6.0

# In the container, go to the OpenPCDet repo
# clean all built openpcdet
python setup.py clean
# build the library of OpenPCDet
python setup.py develop

# goes to the tools directory
cd tools

# put our repo here

# run our code with the MAST
python inference_opt.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml \
    --ckpt ../models/pv_rcnn_8369.pth \
    --data_path semantic_kitti \
    --sampling_method ma_mab \
    --predict_method velocity \
    --sequence_id 00 
```

## Technical Report
## Technical Report
The technical report of MAST can be found in the following link:

[MAST Technical Report](https://github.com/gravesprite/MAST/blob/main/MAST_technical_report)

## Dataset
The datasets can be downloaded from:
#### SemanticKitti
http://semantic-kitti.org/dataset.html#format

#### ONCE
https://once-for-auto-driving.github.io/download.html#downloads

#### SynLiDAR
https://github.com/xiaoaoran/SynLiDAR

## Citation
If you find our work useful, please consider citing:
``` shell
@inproceedings{mast2025,
  title={MAST: Towards Efficient Analytical Query Processing on Point Cloud Data [Technical Report]},
  author = {Jiangneng Li and Haitao Yuan and Gao Cong and Han Mao Kiah and Shuhao Zhang},
  booktitle={Proceedings of the 2025 ACM SIGMOD International Conference on Management of Data},
  year={2025}
}
```