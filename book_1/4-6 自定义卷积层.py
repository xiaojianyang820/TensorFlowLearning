import numpy as np
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = tf.expand_dims(x_train, -1)
x_test = tf.expand_dims(x_test, -1)
y_train = np.float32(tf.keras.utils.to_categorical(y_train, num_classes=10))
y_test = np.float32(tf.keras.utils.to_categorical(y_test, num_classes=10))
batch_size = 10
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size).shuffle(batch_size * 10)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)


class MyLayer(tf.keras.layers.Layer):
    def __init__(self, kernel_size, filter):
        self.filter = filter
        self.kernel_size = kernel_size
        super(MyLayer, self).__init__()

    def build(self, input_shape):
        self.weight = tf.Variable(tf.random.normal([self.kernel_size, self.kernel_size, input_shape[-1], self.filter]))
        self.bias = tf.Variable(tf.random.normal([self.filter]))
        super(MyLayer, self).build(input_shape)

    def call(self, input_tensor):
        conv = tf.nn.conv2d(input_tensor, self.weight, strides=[1, 2, 2, 1], padding='SAME')
        conv = tf.nn.bias_add(conv, self.bias)
        out = tf.nn.relu(conv) + conv
        return out


input_xs = tf.keras.Input([28, 28, 1])
conv = tf.keras.layers.Conv2D(32, 3, padding='SAME', activation=tf.nn.relu)(input_xs)
# 使用自定义的卷积层
conv = MyLayer(32, 3)(conv)
conv = tf.keras.layers.BatchNormalization()(conv)
conv = tf.keras.layers.Conv2D(64, 3, padding='SAME', activation=tf.nn.relu)(conv)
conv = tf.keras.layers.MaxPool2D(strides=[1, 1])(conv)
conv = tf.keras.layers.Conv2D(128, 3, padding='SAME', activation=tf.nn.relu)(conv)
flat = tf.keras.layers.Flatten()(conv)
dense = tf.keras.layers.Dense(512, activation=tf.nn.relu)(flat)
logits = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(dense)
model = tf.keras.Model(inputs=input_xs, outputs=logits)
model.compile(optimizer=tf.optimizers.Adam(0.001), loss=tf.losses.categorical_crossentropy,
              metrics=['accuracy'])
model.fit(train_dataset, epochs=10)
model.save('./saver/model.h5')
score = model.evaluate(test_dataset)
print('Last Score: ', score)

