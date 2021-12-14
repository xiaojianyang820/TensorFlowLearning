import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris


data = load_iris()
iris_target = data.target
iris_data = np.float32(data.data)
iris_target = np.float32(tf.keras.utils.to_categorical(iris_target, num_classes=3))
iris_data = tf.data.Dataset.from_tensor_slices((iris_data, iris_target)).batch(50)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(3, activation='softmax'))

opt = tf.optimizers.Adam(0.001)
for epoch in range(1000):
    for item_data, item_label in iris_data:
        with tf.GradientTape() as tape:
            logits = model(item_data)
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=item_label,
                                                                           y_pred=logits))
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
    print('Training Loss is: ', loss.numpy())
