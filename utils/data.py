import tensorflow as tf
import matplotlib.pyplot as plt

def example_display(dataset_info, dataset, num_examples=16, is_raw = False):
    row_num = num_examples // 8 + 1
    dis = plt.figure(figsize=(15, 2 * (row_num)))
    get_examples_label = dataset_info.features['label'].int2str

    dataset.batch(num_examples)
    for idx, (Image, Label) in enumerate(dataset.take(num_examples)):
        ax = dis.add_subplot(row_num, 8, idx + 1)
        
        if not is_raw:
            Image = (Image + 1) / 2
        
        ax.imshow(Image)
        ax.set_title(get_examples_label(Label))
        ax.axis('off')

def resize_and_rescale(image, label, size = (224, 224)):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, size)
    image = image / 127.5 - 1
    return image, label

def get_batches(train_ds, val_ds, test_ds, batch_size = 64):
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)
    test_ds = test_ds.batch(batch_size)

    return train_ds, val_ds, test_ds