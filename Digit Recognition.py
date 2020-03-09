from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf


# ---------------------------- Define training data -----------------------------------------------------
mnist = tf.keras.datasets.mnist                                         # load data set
(x_train, y_train), (x_test, y_test) = mnist.load_data()                # load data set
x_train, x_test = x_train / 255.0, x_test / 255.0                       # covert x_train and x_test to integers


# ---------------------------- Define a model (network of layers) ---------------------------------------
model = tf.keras.models.Sequential([                                    # linear stack of layers
  tf.keras.layers.Flatten(input_shape=(28, 28)),                        # flattens the input (images of digit are 28 by 28 pixels)
  tf.keras.layers.Dense(128, activation='relu'),                        # fully connected (densely connected) layer (relu is a common activation function)
  tf.keras.layers.Dropout(0.2),                                         # reduce complexity by deactivating random neurons in a layer
  tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()                                # produces logits
predictions

tf.nn.softmax(predictions).numpy()                                      # this function (softmax) that takes logits as inputs and converts them to probabilities


# ---------------------------- Configure the learning process (loss function, optimizer) ----------------
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)   # function that returns a loss for each example (loss function)

loss_fn(y_train[:1], predictions).numpy()                               # use the loss function

model.compile(optimizer='adam',                                         # configure the model for training, using the adam optimizer (an algorithm for gradient-based optimization of stochastic objective functions)
              loss=loss_fn,                                             # use the previously defined loss function
              metrics=['accuracy'])                                     # metric is a function that is used to judge the performance of the model


# ---------------------------- Iterate training data by calling the fit function ------------------------
model.fit(x_train, y_train, epochs=5)                                   # adjust the parameters to minimize the loss


# ---------------------------- Check performance of NN --------------------------------------------------
model.evaluate(x_test,  y_test, verbose=2)                              # checks the performance of the model (evaluates it)

probability_model = tf.keras.Sequential([                               # function that return the probability of the model
  model,
  tf.keras.layers.Softmax()
])

probability_model(x_test[:5])