# %%
import tensorflow as tf
from tensorflow import GradientTape
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from IPython.display import HTML, Markdown

# %%
num_samples_per_class = 1000
negative_samples = np.random.multivariate_normal(
        mean=[0, 3],
        cov=[[1, 0.5], [0.5, 1.]],
        size=num_samples_per_class)
positive_samples = np.random.multivariate_normal(
        mean=[3, 0],
        cov=[[1, .5], [.5, 1.]],
        size=num_samples_per_class)

# stack data points
inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)
# create labels
targets = np.vstack((np.zeros((num_samples_per_class, 1), dtype=np.float32),
                     np.ones((num_samples_per_class, 1), dtype=np.float32)))

# %%
plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
plt.show()

# %%
# Creating Linear Classifier Variables p81 listing 3.17
input_dim = 2
output_dim = 1
W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))

# %%
def model(inputs):
    return tf.matmul(inputs, W) + b

# %% [markdown]
# Because our linear classifier operates on 2D iputes, W is really just two scaler coefficients, w1 and w2: W = [[w1], [w2]].
# Meanwhile b is a single scalar coefficient. As such, for a given input point [x, y], its prediction value is equal to:
#
# $$\large (w_1 \times x_i) + (w_2 \times y_i) + b $$

# %%
def square_loss(targets, predictions):
    per_sample_losses = tf.square(targets - predictions)
    return tf.reduce_mean(per_sample_losses)

# %% [markdown]
# Listing 3.20 "The Training Step Function" 

# %%
learning_rate = 0.1

def training_step(inputs, targets):
    with GradientTape() as tape:
        predictions = model(inputs)
        loss = square_loss(targets, predictions)
    grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b])
    W.assign_sub(grad_loss_wrt_W * learning_rate)
    b.assign_sub(grad_loss_wrt_b * learning_rate)
    return loss

# %%
tab_data = [["step", "loss"]]
for step in range(40):
    loss = training_step(inputs, targets)
    tab_data.append([f"{step}", f"{loss:.4f}"])
Markdown(tabulate(tab_data, headers='firstrow', tablefmt="github"))


# %%
predictions = model(inputs)
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > .5)
plt.show()

# %% [markdown]
# Equation for the classifier line is:
# $$\large y = \dfrac{\dfrac{-w_1 \times x}{w_2} + ( \dfrac{1}{2} - b )}{w_2} $$

# %%
x = np.linspace(-1, 4, 100)
y = -W[0] / W[1] * x + (0.5 - b) / W[1]
plt.plot(x, y, c='red')
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > .5)
plt.show()

# %%


