import tensorflow, keras
from keras import layers

def image_preprocessing_v1():
    model = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.2),
        
        layers.Normalization(),
    ])
    
    model.name = "image_preprocessing_v1"
    
    return model
    
def image_preprocessing_v2():
    model = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.2),
        
        layers.BatchNormalization(),
    ])
    
    model.name = "image_preprocessing_v2"
    
    return model