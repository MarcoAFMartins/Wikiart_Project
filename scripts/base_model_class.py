#Model Class
from keras import Model
import keras

# model building imports
from keras import Model, Sequential, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Rescaling, RandAugment, GlobalAveragePooling2D, GlobalMaxPool2D, Dropout, RandomRotation, RandomFlip, BatchNormalization, Activation
from keras.optimizers import SGD, Adam
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy, AUC, F1Score
from keras.callbacks import EarlyStopping


#Import our custom augmentations
from augmentation import (
    augmentation_moderate
)

@keras.saving.register_keras_serializable()
class BatchNormalization2_Model(Model):
    """
    Adding Batch Normalization Layers (before activation function)

    """
    #Initialization
    def __init__(self, **kwargs):
        super().__init__(name="BatchNormalization_Model2")

        self.n_classes = 23
        self.Rescaling = Rescaling(1./255)
        self.augmentation_layer = augmentation_moderate

        #First Convolutional Block
        self.Conv1 = Conv2D(
            filters=40,
            kernel_size=(3, 3),
            padding= "same",
            name="conv_layer_1"
        )

        self.BatchNormalization_layer1 = BatchNormalization()

        self.activation_1 = Activation(activation="relu")

        #First Pooling Layer
        self.max_pool_layer_1 = MaxPooling2D(
            pool_size=(2, 2),
            name="max_pool_layer_1"
        )


        #Second Convolutional Block
        self.Conv2 = Conv2D(
            filters=40,
            kernel_size=(3, 3),
            padding= "same",
            name="conv_layer_2"
        )

        self.BatchNormalization_layer2 = BatchNormalization()

        self.activation_2 = Activation(activation="relu")

        #Second Pooling Layer
        self.max_pool_layer_2 = MaxPooling2D(
            pool_size=(2, 2),
            name="max_pool_layer_2"
        )

        #Third Convolutional Block
        self.Conv3 = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding= "same",
            name="conv_layer_3"
        )

        self.BatchNormalization_layer3 = BatchNormalization()

        self.activation_3 = Activation(activation="relu")

        #Third Pooling Layer
        self.max_pool_layer_3 = MaxPooling2D(
            pool_size=(2, 2),
            name="max_pool_layer_3"
        )


        #Fourth Convolutional Block
        self.Conv4 = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding= "same",
            name="conv_layer_4"
        )

        self.BatchNormalization_layer4 = BatchNormalization()

        self.activation_4 = Activation(activation="relu")

        #Fourth Pooling Layer
        self.max_pool_layer_4 = MaxPooling2D(
            pool_size=(2, 2),
            name="max_pool_layer_4"
        )

        self.Globalaveragepooling_layer = GlobalAveragePooling2D()
        self.Dense_1 = Dense(100, activation="relu")
        self.Drop_out =Dropout(rate=0.2)
        self.Dense_2 = Dense(self.n_classes, activation="softmax")


    def call(self, inputs):

        x = inputs

        x = self.Rescaling(x)

        
        x = self.augmentation_layer(x)

        x = self.Conv1(x)
        x = self.BatchNormalization_layer1(x)
        x = self.activation_1(x)
        x = self.max_pool_layer_1(x)

        x = self.Conv2(x)
        x = self.BatchNormalization_layer2(x)
        x = self.activation_2(x)
        x = self.max_pool_layer_2(x)

        x = self.Conv3(x)
        x = self.BatchNormalization_layer3(x)
        x = self.activation_3(x)
        x = self.max_pool_layer_3(x)

        x = self.Conv4(x)
        x = self.BatchNormalization_layer4(x)
        x = self.activation_4(x)
        x = self.max_pool_layer_4(x)

        x = self.Globalaveragepooling_layer(x)
        x = self.Dense_1(x)
        x = self.Drop_out(x)
        x = self.Dense_2(x)

        return x