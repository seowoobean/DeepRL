import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential(
    [
        layers.Dense(2, activation="relu"),
        layers.Dense(3, activation="relu"),
        layers.Dense(4),
    ]
)
x = tf.ones((100, 4))
y = model(x)
print("Number of weights after calling the model:", len(model.weights))  # 2*3 = 6

model.summary() # Param # -> Dimension is 4 + bias = 5
                # Param # of Output Shape (100, 2) -> 5*2 = 10 and 2+bias=3
                # Param # of Output Shape (100, 3) -> 3*3 = 9