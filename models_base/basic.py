import tensorflow as tf, keras
from keras import models, layers

def base_simple1():
    model = models.Sequential([
    layers.Conv2D(16, (3, 3), padding = 'same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(32, (3, 3), padding = 'same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), padding = 'same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    ])
    
    model.name = "base_simple1"
    
    return model

def base_simple2(input_shape = (480, 480, 3)):
    
    input = keras.Input(shape=input_shape)
    features = input
    
    for i in range(5):
        features = layers.Conv2D(2 ** (i + 4), (3, 3), padding = 'same', activation='relu')(features)
        features = layers.MaxPooling2D(pool_size=(2, 2))(features)
        
    output = features
        
    model = models.Model(inputs=input, outputs=output)
    model.name = "base_simple2"
    
    return model

def base_simple3(input_shape = (480, 480, 3)):
    
    input = keras.Input(shape=input_shape)
    features = input
    
    for i in range(5):
        features = layers.Conv2D(2 ** (i + 4), (3, 3), padding = 'same', activation='relu')(features)
        features = layers.Conv2D(2 ** (i + 4), (3, 3), padding = 'same', activation='relu')(features)
        features = layers.Conv2D(2 ** (i + 4), (3, 3), padding = 'same', activation='relu')(features)
        features = layers.MaxPooling2D(pool_size=(2, 2))(features)
        
    output = features
        
    model = models.Model(inputs=input, outputs=output)
    model.name = "base_simple2"
    
    return model
    