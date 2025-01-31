import keras

def EV2L(image_shape=(480, 480, 3), top=False, trainable=False):
    model = keras.applications.EfficientNetV2L(
        weights='imagenet',
        include_top=top,
        input_shape=image_shape
    )
    
    model.trainable = trainable
    
    return model