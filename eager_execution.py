import os
import tensorflow as tf
import cProfile
import numpy as np

print(tf.executing_eagerly())
x = [[2.]]
m = tf.matmul(x, x)
print("hello, {}".format(m))
a = tf.constant([[1, 2],
                 [3, 4]])
print(a)
b = tf.add(a, 1)
print(b)
print(a*b)

#Eager traning ---> Automatic differentiation
w = tf.Variable([[1.0]])
with tf.GradientTape() as tape:
  loss = w * w
#tape --> monitering variable
grad = tape.gradient(loss, w)  #w에 대해 loss를 미분한 값  loss = 2w (w=[[1.0]])
print(grad)  # => tf.Tensor([[ 2.]], shape=(1, 1), dtype=float32)