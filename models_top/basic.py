import tensorflow, keras
from keras import layers

def simple1():
    model = keras.Sequential([
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5)
    ])
    
    model.name = "basic_simple1"
    
    return model

def simple2():
    model = keras.Sequential([
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu')
    ])
    
    model.name = "basic_simple2"
    
    return model

def midsize1():
    model = keras.Sequential([
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu')
    ])
    
    model.name = "basic_midsize1"
    
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
    
    model.name = "basic_midsize1"
    
    return model

def midsize3():
    inputs = keras.Input(shape=(7, 7, 1280))
    
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
    
    model = keras.Model(inputs=inputs, outputs=output, name="basic_midsize3")
    return model