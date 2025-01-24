import keras

def EV2L(image_shape=(480, 480, 3), trainable=False):
    efficientNetV2L = keras.applications.EfficientNetV2L(
        weights='imagenet',
        include_top=False,
        input_shape=image_shape
    )
    
    efficientNetV2L.trainable = trainable
    
    return efficientNetV2L