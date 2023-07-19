import os
import numpy as np
import cv2
#from keras.utils import to_categorical


from keras.layers import Input, Dense
from keras.models import Model



# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Numpy-related utilities."""

import numpy as np

# isort: off
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.utils.to_categorical")
def to_categorical(y, num_classes=None, dtype="float32"):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with `categorical_crossentropy`.

    Args:
        y: Array-like with class values to be converted into a matrix
            (integers from 0 to `num_classes - 1`).
        num_classes: Total number of classes. If `None`, this would be inferred
          as `max(y) + 1`.
        dtype: The data type expected by the input. Default: `'float32'`.

    Returns:
        A binary matrix representation of the input as a NumPy array. The class
        axis is placed last.

    Example:

    >>> a = tf.keras.utils.to_categorical([0, 1, 2, 3], num_classes=4)
    >>> print(a)
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]]

    >>> b = tf.constant([.9, .04, .03, .03,
    ...                  .3, .45, .15, .13,
    ...                  .04, .01, .94, .05,
    ...                  .12, .21, .5, .17],
    ...                 shape=[4, 4])
    >>> loss = tf.keras.backend.categorical_crossentropy(a, b)
    >>> print(np.around(loss, 5))
    [0.10536 0.82807 0.1011  1.77196]

    >>> loss = tf.keras.backend.categorical_crossentropy(a, a)
    >>> print(np.around(loss, 5))
    [0. 0. 0. 0.]
    """
    y = np.array(y, dtype="int")
    input_shape = y.shape

    # Shrink the last dimension if the shape is (..., 1).
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])

    y = y.reshape(-1)
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


@keras_export("keras.utils.to_ordinal")
def to_ordinal(y, num_classes=None, dtype="float32"):
    """Converts a class vector (integers) to an ordinal regression matrix.

    This utility encodes class vector to ordinal regression/classification
    matrix where each sample is indicated by a row and rank of that sample is
    indicated by number of ones in that row.

    Args:
        y: Array-like with class values to be converted into a matrix
            (integers from 0 to `num_classes - 1`).
        num_classes: Total number of classes. If `None`, this would be inferred
            as `max(y) + 1`.
        dtype: The data type expected by the input. Default: `'float32'`.

    Returns:
        An ordinal regression matrix representation of the input as a NumPy
        array. The class axis is placed last.

    Example:

    >>> a = tf.keras.utils.to_ordinal([0, 1, 2, 3], num_classes=4)
    >>> print(a)
    [[0. 0. 0.]
     [1. 0. 0.]
     [1. 1. 0.]
     [1. 1. 1.]]
    """
    y = np.array(y, dtype="int")
    input_shape = y.shape

    # Shrink the last dimension if the shape is (..., 1).
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])

    y = y.reshape(-1)
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    range_values = np.arange(num_classes - 1)
    range_values = np.tile(np.expand_dims(range_values, 0), [n, 1])
    ordinal = np.zeros((n, num_classes - 1), dtype=dtype)
    ordinal[range_values < np.expand_dims(y, -1)] = 1
    output_shape = input_shape + (num_classes - 1,)
    ordinal = np.reshape(ordinal, output_shape)
    return ordinal


@keras_export("keras.utils.normalize")
def normalize(x, axis=-1, order=2):
    """Normalizes a Numpy array.

    Args:
        x: Numpy array to normalize.
        axis: axis along which to normalize.
        order: Normalization order (e.g. `order=2` for L2 norm).

    Returns:
        A normalized copy of the array.
    """
    l2 = np.atleast_1d(np.linalg.norm(x, order, axis))
    l2[l2 == 0] = 1
    return x / np.expand_dims(l2, axis)

is_init = False
size = -1

label = []
dictionary = {}
c = 0

for i in os.listdir():
	if i.split(".")[-1] == "npy" and not(i.split(".")[0] == "labels"):  
		if not(is_init):
			is_init = True 
			X = np.load(i)
			size = X.shape[0]
			y = np.array([i.split('.')[0]]*size).reshape(-1,1)
		else:
			X = np.concatenate((X, np.load(i)))
			y = np.concatenate((y, np.array([i.split('.')[0]]*size).reshape(-1,1)))

		label.append(i.split('.')[0])
		dictionary[i.split('.')[0]] = c  
		c = c+1


for i in range(y.shape[0]):
	y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")

###  hello = 0 nope = 1 ---> [1,0] ... [0,1]

y = to_categorical(y)

X_new = X.copy()
y_new = y.copy()
counter = 0 

cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)

for i in cnt: 
	X_new[counter] = X[i]
	y_new[counter] = y[i]
	counter = counter + 1


ip = Input(shape=(X.shape[1]))

m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)

op = Dense(y.shape[1], activation="softmax")(m) 

model = Model(inputs=ip, outputs=op)

model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

model.fit(X, y, epochs=50)


model.save("model.h5")
np.save("labels.npy", np.array(label))
