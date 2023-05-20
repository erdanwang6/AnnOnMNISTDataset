import tensorflow as tf
import numpy as np
import timm

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Reshape the images to have a single channel (grayscale) and pad to be 32x32
train_images = np.pad(train_images, ((0, 0), (2, 2), (2, 2)))[:, :, :, np.newaxis]
test_images = np.pad(test_images, ((0, 0), (2, 2), (2, 2)))[:, :, :, np.newaxis]

# Convert labels to one-hot encoding
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)


def create_vit_model():
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=10,img_size=32)
    return model

vit_model = create_vit_model()

# Compile the model
vit_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

# Train the model
batch_size = 64
epochs = 20
history = vit_model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size,
                        validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_acc = vit_model.evaluate(test_images, test_labels)

print(f'Test loss: {test_loss}, Test accuracy: {test_acc}')
