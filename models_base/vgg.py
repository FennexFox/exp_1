import keras

def VGG16(image_shape=(224, 224, 3), trainable=False):
    efficientNetV2L = keras.applications.VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=image_shape
    )
    
    efficientNetV2L.trainable = trainable
    
    return efficientNetV2L