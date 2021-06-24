import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

image_size = (195, 130)
batch_size = 32
color_mode = "grayscale"
num_training_batches = 6

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "plant_dataset/images",
    validation_split=0.2,
    subset="training",
    seed = 1337,
    color_mode = color_mode,
    image_size=image_size,
    batch_size=batch_size,
)

test_ds = train_ds.take(num_training_batches)
train_ds = train_ds.skip(num_training_batches)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "plant_dataset/images",
    validation_split=0.2,
    subset="validation",
    seed = 1337,
    color_mode = color_mode,
    image_size=image_size,
    batch_size=batch_size,
)

train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)

augmented_train_ds = train_ds.map(
  lambda x, y: (data_augmentation(x, training=True), y))

def make_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 3, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (1,))

epochs = 100

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    augmented_train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)

print(f"testing model on {num_training_batches * batch_size} images")
(test_loss, test_acc) = model.evaluate(test_ds)
print(f"test loss: {test_loss}, accuracy: {test_acc}")
