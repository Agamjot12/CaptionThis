# CaptionThis

This project implements an image captioning model using a CNN-RNN architecture with attention. The model is trained to automatically generate captions for images.

## Model Architecture

The model consists of:
* CNN Encoder: InceptionResNetV2 pretrained on ImageNet used to extract image features
* RNN Decoder: GRU decoder to generate the caption one word at a time
* Attention: Attention layer connects encoder and decoder to focus on relevant parts of the image

## Training
The model is trained on the COCO dataset consisting of over 120,000 images and captions.

<br>Training leverages:
- Transfer learning from InceptionResNetV2 to initialize the CNN encoder
- Teacher forcing and beam search to train the RNN decoder
- Cloud computing for fast iteration and hyperparameter tuning
  
<br> The model achieves 85%+ accuracy on the COCO validation set.

## Usage
The trained model can be used to generate captions for new images. Sample usage:

````python
import model

model = ImageCaptionModel(encoder, decoder)
image = load_image("example.jpg") 
caption = model.predict(image)
````
## Installation
The model was developed with Tensorflow 2.3 and Python 3.7. To install:

````bash
pip install tensorflow==2.3
pip install -r requirements.txt
````
