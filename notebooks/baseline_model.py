import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import keras
from keras import layers

n_classes = 23

class BaselineModel(keras.Model):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.augmentation = layers.Pipeline(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomBrightness(factor=0.08, value_range=(0.0, 1.0)),
                layers.RandomContrast(factor=0.08),
                layers.RandomRotation(factor=0.02, fill_mode="reflect"),
                layers.RandomZoom((-0.05, 0.05), fill_mode="reflect"),
                layers.RandomTranslation(
                    height_factor=0.05, width_factor=0.05, fill_mode="reflect"
                ),
                layers.GaussianNoise(stddev=0.02),
            ],
            name="augmentation_moderate_noise",
        )
        self.rescaling = layers.Rescaling(1./255)
        
        self.conv1 = layers.Conv2D(128, 3, activation="relu")
        self.conv2 = layers.Conv2D(128, 3, activation="relu")
        self.max_pooling1 = layers.MaxPooling2D(2)
        
        self.conv3 = layers.Conv2D(256, 3, activation="relu")
        self.conv4 = layers.Conv2D(256, 3, activation="relu")
        self.max_pooling2 = layers.MaxPooling2D(2)
        
        self.conv5 = layers.Conv2D(256, 3, activation="relu")
        self.conv6 = layers.Conv2D(256, 3, activation="relu")
        self.max_pooling3 = layers.MaxPooling2D(2)
        
        self.conv7 = layers.Conv2D(512, 3, activation="relu")
        self.conv8 = layers.Conv2D(512, 3, activation="relu")
        self.max_pooling4 = layers.MaxPooling2D(2)
        
        self.global_average_pooling = layers.GlobalAveragePooling2D()
        self.dropout = layers.Dropout(0.2)
        self.classifier = layers.Dense(n_classes, activation="softmax")
        
    def call(self, inputs, training=False):
        x = self.rescaling(inputs)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max_pooling1(x)
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pooling2(x)
        
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.max_pooling3(x)
        
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.max_pooling4(x)
        
        x = self.global_average_pooling(x)
        x = self.dropout(x)
        outputs = self.classifier(x)
        return outputs