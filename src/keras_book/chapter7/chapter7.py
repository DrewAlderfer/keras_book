import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

config = {"vocabulary_size": 10000,
          "num_tags": 100,
          "num_departments": 4,
          "num_samples": 1280}

feature_data = {"title_data": np.random.randint(0, 2, size=(config['num_samples'], config['vocabulary_size'])),
                "text_body_data": np.random.randint(0, 2, size=(config['num_samples'], config['vocabulary_size'])),
                "tags_data": np.random.randint(0, 2, size=(config['num_samples'], config['num_tags']))}

target_data = {"priority_data": np.random.random(size=(config['num_samples'], 1)),
               "deparment_data": np.random.randint(0, 2, size=(config['num_samples'], config['num_departments']))}

def functional_api_model(vocabulary_size, num_tags, num_departments, num_samples):

    # Feature Data
    title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
    text_body_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
    tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))

    # Target Data
    priority_data = np.random.random(size=(num_samples, 1))
    deparment_data = np.random.randint(0, 2, size=(num_samples, num_departments))

    # Input Instantiation
    title = tf.keras.Input(shape=(vocabulary_size,), name='title')
    text_body = tf.keras.Input(shape=(vocabulary_size,), name="text_body")
    tags = tf.keras.Input(shape=(num_tags,), name='tags')

    # Layer Instantiation
    features = tf.keras.layers.Concatenate()([title, text_body, tags])
    features = tf.keras.layers.Dense(64, activation='relu')(features)
    # Output Layers
    priority = tf.keras.layers.Dense(1, activation="sigmoid", name="priority")(features)
    department = tf.keras.layers.Dense(
            num_departments, activation="softmax", name="department")(features)
            
    # Instantiate Model
    model = tf.keras.Model(inputs=[title, text_body, tags],
                        outputs=[priority, department])
    model.compile(optimizer="rmsprop",
                  loss=["mean_squared_error", "categorical_crossentropy"],
                  metrics=[["mean_absolute_error"], ["accuracy"]])
    model.fit([title_data, text_body_data, tags_data],
              [priority_data, deparment_data],
               epochs=1)
    priority_preds, department_preds = model.predict(
            [title_data, text_body_data, tags_data])
 
    return model, priority_preds, department_preds



class CustomerTicketModel(tf.keras.Model):

    def __init__(self, num_departments):
        super().__init__()
        self.concat_layer = tf.keras.layers.Concatenate()
        self.mixing_layer = tf.keras.layers.Dense(64, activation="relu")
        self.priority_scorer = tf.keras.layers.Dense(1, activation="sigmoid")
        self.department_classifier = tf.keras.layers.Dense(
                num_departments, activation="softmax")
                
    def call(self, inputs):
        title = inputs["title"]
        text_body = inputs["text_body"]
        tags = inputs["tags"]

        features = self.concat_layer([title, text_body, tags])
        features = self.mixing_layer(features)
        priority = self.priority_scorer(features)
        department = self.department_classifier(features)
        return priority, department


class RootMeanSquaredError(tf.keras.metrics.Metric):

    def __init__(self, name="rmse", **kwargs):
        print("hello 2")
        super().__init__(name=name, **kwargs)
        self.mse_sum = self.add_weight(name="mse_sum", initializer="zeros")
        self.total_samples = self.add_weight(
                name="total_samples", initializer="zeros", dtype="int32")

    def update_state(self, y_true, y_pred, **kwargs):
        y_true = tf.one_hot(y_true, depth=tf.shape(y_pred)[1])
        mse = tf.reduce_sum(tf.square(y_true - y_pred))
        self.mse_sum.assign_add(mse)
        num_samples = tf.shape(y_pred)[0]
        self.total_samples.assign_add(num_samples)

    def result(self):
        return tf.sqrt(self.mse_sum / tf.cast(self.total_samples, tf.float32))

    def reset_state(self):
        self.mse_sum.assign_add(0.)
        self.total_samples.assign_add(0)

class LossHistory(tf.keras.callbacks.Callback):

    def on_train_begin(self, logs):
        self.per_batch_losses = []

    def on_batch_end(self, batch, logs=None):
        self.per_batch_losses.append(logs.get("loss"))

    def on_epoch_end(self, epoch, logs):
        plt.clf()
        plt.plot(range(len(self.per_batch_losses)), self.per_batch_losses,
                 label="Training loss for each batch")
        plt.xlabel(f"Batch (epoch {epoch})")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"./chapter7/images/fig7_20_epoch_{epoch}.png")
        self.per_batch_losses = []

    def on_train_end(self, logs=None):
        plt.close()


def training_loop_ex(model, train_data, train_labels, 
                     loss = tf.keras.losses.SparseCategoricalCrossentropy(),
                     epochs=3,
                     validation_data:tuple=(None, None),
                     optimizer = tf.keras.optimizers.RMSprop(),
                     metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
                     ):
    
    loss_fn = loss 
    loss_tracking_metric = tf.keras.metrics.Mean()
    val_data, val_labels = validation_data
    logs = {}

    def training_loop(logs=logs):
        training_dataset = tf.data.Dataset.from_tensor_slices(
                (train_data, train_labels))
        training_dataset = training_dataset.batch(32)

        for epoch in range(epochs):
            reset_metrics()
            for inputs_batch, targets_batch in training_dataset:
                logs = train_step(inputs_batch, targets_batch)
            print(f"Results at the end of epoch {epoch}")
            for key, value in logs.items():
                print(f"...{key}: {value}:.4f")

    def validation_loop(logs=logs):
        val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
        val_dataset = val_dataset.batch(32)
        reset_metrics()
        for inputs_batch, targets_batch in val_dataset:
            logs = test_step(inputs_batch, targets_batch)
        print("Evaluation results:")
        for key, value in logs.items():
            print(f"...{key}: {value:.4f}")

    @tf.function
    def train_step(inputs, targets):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = loss_fn(targets, predictions)
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        logs = {}
        for metric in metrics:
            metric.update_state(targets, predictions)
            logs[metric.name] = metric.result()

        loss_tracking_metric.update_state(loss)
        logs['loss'] = loss_tracking_metric.result()
        return logs

    def reset_metrics():
        for metric in metrics:
            metric.reset_state()
        loss_tracking_metric.reset_state()


    def test_step(inputs, targets):
        predictions = model(inputs, training=False)
        loss = loss_fn(targets, predictions)

        logs = {}
        for metric in metrics:
            metric.update_state(targets, predictions)
            logs["val_" + metric.name] = metric.result()
        
        loss_tracking_metric.update_state(loss)
        logs['val_loss'] = loss_tracking_metric.result()
        return logs

    training_loop()
    validation_loop()


class CustomModel(tf.keras.Model):
    def train_step(self, data):
        inputs, targets = data
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = self.compiled_loss(targets, predictions)
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.compiled_metrics.update_state(targets, predictions)
        return {m.name: m.result() for m in self.metrics}
