# Densely Connected Convolutional Network (DenseNet)
paper: [Densely Connected Convolutional Networks](http://arxiv.org/abs/1608.06993)

**This is the MXNet implement of DenseNet with pretrained model, therefor you can fine-tune in the pretrained model for your own dataset.**

## Do as follows:

* Download pretrained models(pretrained in ImageNet dataset) from

|Network 			   | Top-1 error |     MXNet model|
|:-------------------: | :----------:|:--------------:| 
|DenseNet-121 (k=32)   |	25.16  	 |[Google Drive (32.3MB)](https://drive.google.com/open?id=0ByXcv9gLjrVcb3NGb1JPa3ZFQUk)|
|DenseNet-169 (k=32)   |	23.74	 |[Google Drive (57.3MB)](https://drive.google.com/open?id=0ByXcv9gLjrVcOWZJejlMOWZvZmc)|
|DenseNet-201 (k=32)   |	22.54	 |[Google Drive (81.0MB)](https://drive.google.com/open?id=0ByXcv9gLjrVcUjF4MDBwZ3FQbkU)|
|DenseNet-161 (k=48)   |	22.28	 |[Google Drive (115.7MB)](https://drive.google.com/open?id=0ByXcv9gLjrVcS0FwZ082SEtiUjQ)|

These pretrained models are manually converted from https://github.com/shicai/DenseNet-Caffe ,put the pretrained model under `/DenseNet/model/` file.

* I produce two ways of image data reading:

**If you want to use `.rec` file to train your model:**

* Change some configuration in `run_train_rec.sh`, for example: `--epoch` and `--model` are corresponding to the pretrained model, `--data-train` is your train `.rec` file, `--save-result` is the train result you want to save, `--num-examples` is the number of your training data, `--save-name` is the name of final model.
* Run
```
sh run_train_rec.sh
```

**If you want to use `.lst` file and image to train your model:**

* Change some configuration in `run_train_lst.sh`, for example: `--epoch` and `--model` are corresponding to the pretrained model, `--data-train` is your train `.lst` file, `--image-train` is your train image file, `--save-result` is the train result you want to save, `--num-examples` is the number of your training data, `--save-name` is the name of final model.
* Run
```
sh run_train_lst.sh
```