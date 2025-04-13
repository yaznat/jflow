# JFlow
CPU based machine learning library in Java.  
High school project.  
Demo files included. Full documentation soon.  
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
##  -> Currently supported activations:  
     - ReLU  
     - LeakyReLU  
     - Softmax  
     - Sigmoid  
     - Tanh  
     - Custom Activation (Easy to implement)  
##  -> Utils  
     - Plot images, confusion matrixes, and more.  
##  -> JMatrix data type  
     - Custom data type to aid in low level use cases.  
     - Stores dimensional information and provides statistics.  
     - Offers operations such as transpose2D and broadcastable mathmatic functions.  