import pickle
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.keras.datasets.cifar import load_batch


def get_cifar100_train_data_and_label(data_path):
    with open(data_path, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    feature = data['data']
    label = data['fine_labels']
    return feature, np.array(label, dtype=int)


def identity_block(input_tensor, out_dim):
    conv1 = tf.keras.layers.Conv2D(out_dim // 4, kernel_size=1, padding='SAME',
                                   activation=tf.nn.relu)(input_tensor)
    conv2 = tf.keras.layers.BatchNormalization()(conv1)
    conv3 = tf.keras.layers.Conv2D(out_dim // 4, kernel_size=3, padding='SAME',
                                   activation=tf.nn.relu)(conv2)
    conv4 = tf.keras.layers.BatchNormalization()(conv3)
    conv5 = tf.keras.layers.Conv2D(out_dim, kernel_size=1, padding='SAME')(conv4)
    out = tf.keras.layers.Add()([input_tensor, conv5])
    out = tf.nn.relu(out)
    return out


def resnet_Model(n_dim=10):
    input_xs = tf.keras.Input(shape=[32, 32, 3])
    conv_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='SAME', activation=tf.nn.relu)(input_xs)
    # +++++++ 第一层 ++++++++++++
    out_dim = 64
    identity_1 = tf.keras.layers.Conv2D(filters=out_dim, kernel_size=3, padding='SAME', activation=tf.nn.relu)(conv_1)
    identity_1 = tf.keras.layers.BatchNormalization()(identity_1)
    for _ in range(3):
        identity_1 = identity_block(identity_1, out_dim)
    # +++++++ 第二层 ++++++++++++
    out_dim = 28
    identity_2 = tf.keras.layers.Conv2D(filters=out_dim, kernel_size=3, padding='SAME',
                                        activation=tf.nn.relu)(identity_1)
    identity_2 = tf.keras.layers.BatchNormalization()(identity_2)
    for _ in range(3):
        identity_2 = identity_block(identity_2, out_dim)
    # +++++++ 第三层 ++++++++++++
    out_dim = 56
    identity_3 = tf.keras.layers.Conv2D(filters=out_dim, kernel_size=3, padding='SAME',
                                        activation=tf.nn.relu)(identity_2)
    identity_3 = tf.keras.layers.BatchNormalization()(identity_3)
    for _ in range(3):
        identity_3 = identity_block(identity_3, out_dim)
    # +++++++ 第四层 ++++++++++++
    out_dim = 52
    identity_4 = tf.keras.layers.Conv2D(filters=out_dim, kernel_size=3, padding='SAME',
                                        activation=tf.nn.relu)(identity_3)
    identity_4 = tf.keras.layers.BatchNormalization()(identity_4)
    for _ in range(1):
        identity_4 = identity_block(identity_4, out_dim)

    flat = tf.keras.layers.Flatten()(identity_4)
    flat = tf.keras.layers.Dropout(0.217)(flat)
    dense = tf.keras.layers.Dense(2048, activation=tf.nn.relu)(flat)
    dense = tf.keras.layers.BatchNormalization()(dense)
    logits = tf.keras.layers.Dense(100, activation=tf.nn.softmax)(dense)
    model = tf.keras.Model(inputs=input_xs, outputs=logits)
    return model


if __name__ == '__main__':
    train_data_path = './data/cifar-100-python/train'
    test_data_path = './data/cifar-100-python/test'
    train_feature, train_label = get_cifar100_train_data_and_label(train_data_path)
    test_feature, test_label = get_cifar100_train_data_and_label(test_data_path)

    path = './data/cifar-100-python'
    fpath = os.path.join(path, 'train')
    x_train, y_train = load_batch(fpath, label_key='fine_labels')
    fpath = os.path.join(path, 'test')
    x_test, y_test = load_batch(fpath, label_key='fine_labels')
    x_train = tf.transpose(x_train, [0, 2, 3, 1])
    y_train = np.float32(tf.keras.utils.to_categorical(y_train, num_classes=100))
    x_test = tf.transpose(x_test, [0, 2, 3, 1])
    y_test = np.float32(tf.keras.utils.to_categorical(y_test, num_classes=100))
    batch_size = 32
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(batch_size*10).\
        batch(batch_size).repeat(3)

    model = resnet_Model()
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    model.fit(train_data, epochs=10)
    score = model.evaluate(x_test, y_test)
    print('Last Score: ', score)




