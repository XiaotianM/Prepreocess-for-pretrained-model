# Prepreocess-for-pretrained-model
Prepreocessing Image for pretrained model for tensorflow, keras, pytorch

## Tensorflow-slim model

slim: https://github.com/tensorflow/models/tree/master/research/slim

slim preprocessing: https://github.com/tensorflow/models/tree/master/research/slim/preprocessing

```python
  preprocessing_fn_map = {
      'cifarnet': cifarnet_preprocessing,
      'inception': inception_preprocessing,
      'inception_v1': inception_preprocessing,
      'inception_v2': inception_preprocessing,
      'inception_v3': inception_preprocessing,
      'inception_v4': inception_preprocessing,
      'inception_resnet_v2': inception_preprocessing,
      'lenet': lenet_preprocessing,
      'mobilenet_v1': inception_preprocessing,
      'mobilenet_v2': inception_preprocessing,
      'mobilenet_v2_035': inception_preprocessing,
      'mobilenet_v2_140': inception_preprocessing,
      'nasnet_mobile': inception_preprocessing,
      'nasnet_large': inception_preprocessing,
      'pnasnet_mobile': inception_preprocessing,
      'pnasnet_large': inception_preprocessing,
      'resnet_v1_50': vgg_preprocessing,
      'resnet_v1_101': vgg_preprocessing,
      'resnet_v1_152': vgg_preprocessing,
      'resnet_v1_200': vgg_preprocessing,
      'resnet_v2_50': vgg_preprocessing,
      'resnet_v2_101': vgg_preprocessing,
      'resnet_v2_152': vgg_preprocessing,
      'resnet_v2_200': vgg_preprocessing,
      'vgg': vgg_preprocessing,
      'vgg_a': vgg_preprocessing,
      'vgg_16': vgg_preprocessing,
      'vgg_19': vgg_preprocessing,
  }
```

Note that the VGG and ResNet V1 parameters have been converted from their original
caffe formats
([here](https://github.com/BVLC/caffe/wiki/Model-Zoo#models-used-by-the-vgg-team-in-ilsvrc-2014)
and
[here](https://github.com/KaimingHe/deep-residual-networks)),
whereas the Inception and ResNet V2 parameters have been trained internally at
Google. `Also be aware that these accuracies were computed by evaluating using a
single image crop. Some academic papers report higher accuracy by using multiple crops at multiple scales.`

about implement details about vgg_preprocessing and inception proprecessing can be find https://blog.csdn.net/shwan_ma/article/details/88708307

### vgg_preprocessing
```python
def preprocess_image(image, output_height, output_width, is_training=False,
                     resize_side_min=_RESIZE_SIDE_MIN,
                     resize_side_max=_RESIZE_SIDE_MAX):
```
- preprocess_for_train  
- preprocess_for_eval

### Inception_preprocessing
```python
def preprocess_image(image, height, width,
                     is_training=False,
                     bbox=None,
                     fast_mode=True,
                     add_image_summaries=True):
```
- preprocess_for_train  
- preprocess_for_eval


## Keras model
https://github.com/keras-team/keras-applications/tree/master/keras_applications

Note: preprocessing methods for keras model have three different ways.
1. tf
```python
    if mode == 'tf':
        x /= 127.5
        x -= 1.
    return x
```
2. caffe
```
    RGB -> BGR
    mean = [103.939, 116.779, 123.68]
    std = None
    x - mean
```
3. torch
```
    x /= 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    (x - mean) / std
```


## pytorch model
https://pytorch.org/docs/stable/torchvision/models.html

`All pre-trained models expect input images normalized in the same way`, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. You can use the following transform to normalize:

```python
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
```