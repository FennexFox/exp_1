import tensorflow, keras, math
from keras import layers

def simple1():
    model = keras.Sequential([
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
    ])
    
    model.name = "top_simple1"
    
    return model

def simple2():
    model = keras.Sequential([
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu')
    ])
    
    model.name = "top_simple2"
    
    return model

def midsize1():
    model = keras.Sequential([
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu')
    ])
    
    model.name = "top_midsize1"
    
    return model

def midsize2():
    model = keras.Sequential([
        layers.GlobalAveragePooling2D(),
        
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(256, activation='relu')
    ])
    
    model.name = "top_midsize1"
    
    return model

def midsize3(input_shape = (480, 480, 1280)):
    conv_output_shape = math.ceil(15 / 480 * input_shape[0])
    
    inputs = keras.Input(shape=(conv_output_shape, conv_output_shape, input_shape[2]))
    
    x = layers.GlobalAveragePooling2D()(inputs)
    x = layers.Dense(1024, activation='relu')(x)

    residual = x
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.add([x, residual])
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    residual = x
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.add([x, residual])
    x = layers.Dense(256, activation='relu')(x)
    output = layers.BatchNormalization()(x)
    
    model = keras.Model(inputs=inputs, outputs=output, name="top_midsize3")
    return model