import tensorflow as tf


y_true = [1, 2]
y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]

scce = tf.keras.losses.SparseCategoricalCrossentropy()
print('\nSparse Categorical Crossentropy')
print(scce(y_true, y_pred))

from tensorflow.keras.losses import sparse_categorical_crossentropy

scce = sparse_categorical_crossentropy
print('\nSparse Categorical Crossentropy')
print(scce(y_true, y_pred))

y_true = [[0.,1.,0.], [0.,0.,1.]]
y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]

cce = tf.keras.losses.CategoricalCrossentropy()
print('\ntf.keras.losses.CategoricalCrossentropy()')
print(cce(y_true, y_pred))


bce = tf.keras.losses.BinaryCrossentropy()
print('\ntf.keras.losses.BinaryCrossentropy()')
print(bce(y_true, y_pred))

bce = tf.keras.losses.BinaryCrossentropy(
    reduction=tf.keras.losses.Reduction.NONE)
print('\ntf.keras.losses.BinaryCrossentropy(): no reduction')
print(bce(y_true, y_pred))

def inverse_sigmoid(x):
    return tf.math.log(x/(1-x))

y_true = [[0.,1.,0.], [0.,0.,1.]]
y_pred = inverse_sigmoid(tf.constant([[0.05, 0.95, 0], [0.1, 0.8, 0.1]], tf.float32))

print('\ntf.nn.sigmoid_cross_entropy_with_logits')
print(tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred))


def binary_cross_entropy(labels, logits):
    epsilon = 1e-7
    logits = tf.clip_by_value(logits, epsilon, 1 - epsilon)
    return -(labels * tf.math.log(logits) +
             (1 - labels) * tf.math.log(1 - logits))

print('\nself defined cross entropy')
print(binary_cross_entropy(
    tf.constant(y_true,tf.float32),
    tf.constant([[0.05, 0.95, 0], [0.1, 0.8, 0.1]], tf.float32)))