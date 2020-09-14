import tensorflow as tf
import numpy as np

#rank = 0
rank_0_tensor = tf.constant(4)
print(rank_0_tensor)
print("===========================================")

#rank = 1
rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
print(rank_1_tensor)
print("===========================================")

#rank = 2
rank_2_tensor = tf.constant([[1, 2],
                             [3, 4],
                             [5, 6]], dtype=tf.float16)
print(rank_2_tensor)
print("===========================================")

#rank = 3
rank_3_tensor = tf.constant([
  [[0, 1, 2, 3, 4],
   [5, 6, 7, 8, 9]],
  [[10, 11, 12, 13, 14],
   [15, 16, 17, 18, 19]],
  [[20, 21, 22, 23, 24],
   [25, 26, 27, 28, 29]],])
print(rank_3_tensor)
print("===========================================")

#numpy 배열로 변경
np.array(rank_2_tensor)
print("===========================================")

#Matrix math
a = tf.constant([[1, 2],
                 [3, 4]])
b = tf.constant([[1, 1],
                 [1, 1]]) # Could have also said `tf.ones([2,2])`
print(tf.add(a, b), "\n")    #element-wise addition
print(tf.multiply(a, b), "\n")   #element-wise multiplication
print(tf.matmul(a, b), "\n")   #Matrix Multiplication a@b
print("===========================================")

#Operations
c = tf.constant([[4.0, 5.0], [10.0, 1.0]])
# Find the largest value
print(tf.reduce_max(c))
# Find the index of the largest value
print(tf.argmax(c))
# Compute the softmax
print(tf.nn.softmax(c))
print(np.exp(4)/(np.exp(4) + np.exp(5)))   #Softmax 계산방법
print("===========================================")

#Shape
rank_4_tensor = tf.zeros([3, 2, 4, 5])
print("Type of every element:", rank_4_tensor.dtype)
print("Number of dimensions:", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
print("Elements along the last axis of tensor:", rank_4_tensor.shape[-1])
print("Total number of elements (3*2*4*5): ", tf.size(rank_4_tensor).numpy())
print("===========================================")

#Manipulating Shape
# Shape returns a `TensorShape` object that shows the size on each dimension
var_x = tf.Variable(tf.constant([[1], [2], [3]]))
print(var_x.shape)
# You can convert this object into a Python list, too
print(var_x.shape.as_list())
# We can reshape a tensor to a new shape.
# Note that we're passing in a list
reshaped = tf.reshape(var_x, [1, 3])
print(var_x.shape)
print(reshaped.shape)
# A `-1` passed in the `shape` argument says "Whatever fits".
print(tf.reshape(rank_3_tensor, [-1]))
print(tf.reshape(rank_3_tensor, [3*2, 5]), "\n")
print(tf.reshape(rank_3_tensor, [3, -1]))
print("===========================================")