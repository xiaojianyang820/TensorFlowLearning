import tensorflow as tf


if __name__ == '__main__':
    print(tf.__version__)
    print(tf.test.is_gpu_available())