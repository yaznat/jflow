# JFlow
CPU based machine learning library in Java.  
High school project.  
Demo files included. Full documentation soon.  
# Strengths  
## -> Memory Optimization  
     - Option of low memory mode for unlimited dataset size.  
     - Train large models with limited resources:  
          - Cifar10 CNN, 1M parameters: < 2G RAM (with low memory mode)  
          - Cifar10 CNN, 5M paramters: < 3G RAM (with low memory mode)  
## -> Low level control and debugging  
     - A clean Keras-similar UI provides the desired level of control over training.  
     - Custom train steps are easy to implement.  
     - Debug mode allows inspection of model gradients.  
# Key Features  
##  -> Dataloader  
     - Easily load images from csv or directory.  
     - Provides useful features such as train-test-split and data batching.  
##  -> Transform  
     - Normalize and augment images with a variety of built-in functions.  
##  -> Sequential  
     - Build models with a simple UI.  
     - High level functions: train, predict.  
     - Low level functions: forward(data) and backward(data) for complex train steps.  
     - Save and load weights.  
##  -> Currently supported layers:  
     - Dense  
     - Conv2D   
     - MaxPool2D  
     - Upsampling2D  
     - Reshape  
     - BatchNorm  
     - Flatten
     - GlobalAveragePooling2D
##  -> Currently supported activation functions:  
     - ReLU  
     - LeakyReLU  
     - Softmax  
     - Sigmoid  
     - Tanh  
     - Swish
     - Mish
     - Custom Activation (Easy to implement)  
## -> Currently supported optimizers:
     - SGD
     - AdaGrad
     - RMSprop
     - Adam
##  -> Utils  
     - Plot images, confusion matrixes, and more.  
##  -> JMatrix data type  
     - Custom data type to aid in low level use cases.  
     - Stores dimensional information and provides statistics.  
     - Offers operations such as transpose2D and broadcastable mathmatic functions.  