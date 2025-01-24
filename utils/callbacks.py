import tensorflow, keras

def callback_savemodel(model_name):
    return keras.callbacks.ModelCheckpoint(
        model_name,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        verbose=0
    )

def callback_earlystop(patience):
    return keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
    )