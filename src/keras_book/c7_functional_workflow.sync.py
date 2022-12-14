# %%
import os
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from chapter7.chapter7 import RootMeanSquaredError, LossHistory,training_loop_ex, CustomModel


# %%
# %load_ext autoreload
# %autoreload 2


# %%
def get_mnist_model():
    inputs = tf.keras.Input(shape=(28 * 28,))
    features = tf.keras.layers.Dense(512, activation="relu")(inputs)
    features = tf.keras.layers.Dropout(0.5)(features)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(features)
    model = tf.keras.Model(inputs, outputs)
    return model

# %%
(images, labels), (test_images, test_labels) = mnist.load_data()
images = images.reshape((60000, 28 * 28)).astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255
train_images, val_images = images[10000:], images[:10000]
train_labels, val_labels = labels[10000:], labels[:10000]


# %%
model = get_mnist_model()
model.compile(optimizer="rmsprop",
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])

# %%
os.environ["XLA_FLAGS"]="--xla_gpu_cuda_data_dir=/home/drew/conda/envs/tf_env/"
model.fit(train_images, train_labels,
          epochs=3,
          validation_data=(val_images, val_labels))

# %%
test_metrics = model.evaluate(test_images, test_labels)
predictions = model.predict(test_images)

# %%
model = get_mnist_model()
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy", RootMeanSquaredError()])
history = model.fit(train_images, train_labels,
          epochs=3,
          validation_data=(val_images,val_labels))
test_metrics = model.evaluate(test_images, test_labels)

# %%
history.history

# %%
model = get_mnist_model()
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy", RootMeanSquaredError()])
history = model.fit(train_images, train_labels,
                    epochs=3,
                    callbacks=[LossHistory()],
                    validation_data=(val_images,val_labels))

# %% [markdown]
# <img src="./chapter7/images/fig7_20_epoch_0.png"></img>

# %%
model_2 = get_mnist_model()
training_loop_ex(model_2, train_images, train_labels,
                 validation_data=(val_images, val_labels))

# %%
inputs = tf.keras.Input(shape=(28 * 28,))
features = tf.keras.layers.Dense(512, activation="relu")(inputs)
features = tf.keras.layers.Dropout(0.5)(features)
outputs = tf.keras.layers.Dense(10, activation="softmax")(features)

model = CustomModel(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
               loss=tf.keras.losses.SparseCategoricalCrossentropy(),
               metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# %%
model.fit(train_images, train_labels, epochs=3)
