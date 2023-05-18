# Segment human using SAM + YOLOv7

This repo implement a interactive tool to segment human by simple clicks on the image, with the combination of 
[YOLOv7](https://github.com/WongKinYiu/yolov7/tree/main) and 
[Segment Anything](https://github.com/facebookresearch/segment-anything)

## Installation
Install the requirements from [YOLOv7](https://github.com/WongKinYiu/yolov7/tree/main) and 
[Segment Anything](https://github.com/facebookresearch/segment-anything)

Install detectron2 from [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

> `git clone https://github.com/facebookresearch/detectron2.git`

Installing the repository

> `python -m pip install -e detectron2`

For windows user, please refer to [this issue](https://github.com/facebookresearch/detectron2/issues/1601) if you have trouble in installing detectron2.




## pretrained models
Download the yolov7 models and put them into  `./checkpoints/yolo/`
* [yolov7-w6-pose.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt)

* [yolov7-mask.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-mask.pt)

Download the SAM models and put them into  `./checkpoints/SAM/`

* [ViT-L SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) is tested, which requires GPU memory >= 8G 
* [ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) 
* [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)


## Citation
If you find this repo helpful, please consider to cite the following papers.

```
@article{wang2022yolov7,
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2207.02696},
  year={2022}
}

@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}

```

## Acknowledgements

* [https://github.com/WongKinYiu/yolov7/tree/main](https://github.com/WongKinYiu/yolov7/tree/main)
* [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
</details>
