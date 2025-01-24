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

def plot_prediction(model, test, batch_size = 32):
    test_batch = test.shuffle(1024).batch(batch_size)
    
    for image_batch, label_batch in test_batch.take(1):
        images = image_batch
        labels = label_batch
        predictions = model.predict(image_batch)
    
    predictions = tf.argmax(predictions, axis=1)
    row_num = batch_size // 8 + 1
    dis = plt.figure(figsize=(15, 2 * (row_num)))
    
    correct_count = 0
    for idx, (image, label, prdiction) in enumerate(zip(images, labels, predictions)):
        ax = dis.add_subplot(row_num, 8, idx + 1)
        image = (image + 1) / 2
        ax.imshow(image)
        
        is_correct = label == prdiction
        title = f'Label: {label},\nPrediction: {prdiction}'
        
        if not is_correct:
            ax.set_title(title, fontdict={'fontsize': 8, 'color': 'red'})
        else:
            ax.set_title(title, fontdict={'fontsize': 8, 'color': 'green'})
            correct_count += 1

        plt.axis('off')
    
    plt.show()
    print(f'Correct Count: {correct_count}/{batch_size}')