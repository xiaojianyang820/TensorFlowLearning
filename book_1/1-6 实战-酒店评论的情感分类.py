import numpy as np
import tensorflow as tf


labels = []
vocab = set()
context = []

with open('data/ChnSentiCorp.txt', 'r', encoding='UTF-8') as emotion_file:
    for line in emotion_file.readlines():
        line = line.strip().split(',')
        labels.append(int(line[0]))
        text = line[1]
        context.append(text)
        for char in text:
            vocab.add(char)

vocab_list = list(sorted(vocab))
print(len(vocab_list))

token_list = []
for text in context:
    token = [vocab_list.index(char) for char in text]
    token = token[:80] + [0] * (80 - len(token))
    token_list.append(token)
token_list = np.array(token_list)
labels = np.array(labels)

input_token = tf.keras.Input(shape=(80, ))
embedding = tf.keras.layers.Embedding(input_dim=3508, output_dim=128)(input_token)
embedding = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128))(embedding)
output = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(embedding)
model = tf.keras.Model(input_token, output)
model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
model.fit(token_list, labels, epochs=10, verbose=2)
