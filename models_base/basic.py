import tensorflow as tf
from keras import models, layers

def simple1(learning_rate=1e-3):
    model = models.Sequential([
    layers.Conv2D(16, (3, 3), padding = 'same', activation='relu', input_shape=(240, 240, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(32, (3, 3), padding = 'same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), padding = 'same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dense(5, activation='softmax')  
    ])
    
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
    
    return model