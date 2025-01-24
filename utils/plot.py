import tensorflow as tf
from matplotlib import pyplot as plt

# designing utility functions
def plot_history(history, loss_ylim=(0, 5), acc_ylim=(0, 1)):
    fig = plt.figure(figsize=(10, 5))
    
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(history.history['loss'], label='train')
    ax1.plot(history.history['val_loss'], label='valid')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_ylim(loss_ylim)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(history.history['accuracy'], label='train')
    ax2.plot(history.history['val_accuracy'], label='valid')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(acc_ylim)
    

    ax1.legend()
    ax2.legend()

    plt.show()