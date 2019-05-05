# cervical_abnormality
This implementation of faster-rcnn is forked from [chenyuntc](https://github.com/chenyuntc/simple-faster-rcnn-pytorch)
with some small bugs fixed, and some codes about dataset have been adapted to this project. 

To use this code please follow instructions by [chenyuntc](https://github.com/chenyuntc/simple-faster-rcnn-pytorch).

## Make our own VOC dataset
To make our own dataset, three python files are needed: 
xml_generator.ipynb : to changed annotations from json file to xml format
split_txt_generator.ipynb : to split dataset into trainval and test
JPEGImages_generator.ipynb : to select images with annotations to train(in the implementation by chenyuntc, train images must have bbox annotation because each train example needs to generate 128 positive anchors)

This is an example of dataset file, images and labels are used to store origin data, Annotations,JPEGImages and ImageSets are generated VOC dataset.
![images](https://github.com/leondelee/cervical_abnormality/blob/detection/simple-faster-rcnn/misc/voc_file_format.png)

## To train
lr has been changed from 1e-3 to 1e-5 to avoid some errors, some other values can be tried.
```
nohup python -m visdom.server -p 5904 &
python train.py train --env='fasterrcnn-caffe' --plot-every=100 --caffe-pretrain –-port=5904 –-voc_data_dir=’home/antoine/cervical_data/cervix_detection_dataset’
```
