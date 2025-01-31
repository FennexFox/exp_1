import keras

def VGG16(image_shape=(224, 224, 3), top=False, trainable=False):
    model = keras.applications.VGG16(
        weights='imagenet',
        include_top=top,
        input_shape=image_shape
    )
    
    model.trainable = trainable
    
    return model