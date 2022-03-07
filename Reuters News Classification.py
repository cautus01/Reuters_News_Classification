# LSTM, CNN, Transformer를 사용한 reuter 뉴스 분류


# 1. 데이터의 전 처리 및 데이터 구조 확인
# 코드 작성자 : 송효주
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import reuters

(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=None, test_split=0.2)

print('훈련용 뉴스 기사 : {}'.format(len(X_train)))
print('테스트용 뉴스 기사 : {}'.format(len(X_test)))
num_classes = len(set(y_train))
print('카테고리 : {}'.format(num_classes))

print('데이터 타입 : {}'.format(type(X_train)))
print('데이터 타입 : {}'.format(type(y_train)))

print(X_train[0]) # 첫번째 훈련용 뉴스 기사
print(y_train[0]) # 첫번째 훈련용 뉴스 기사의 레이블

print('뉴스 기사의 최대 길이 :{}'.format(max(len(l) for l in X_train)))
print('뉴스 기사의 최소 길이 :{}'.format(min(len(l) for l in X_train)))
print('뉴스 기사의 평균 길이 :{}'.format(sum(map(len, X_train))/len(X_train)))

plt.figure(figsize=(16,8))
plt.hist([len(s) for s in X_train], bins=30)
plt.xticks([i for i in range(0,2400,100)])
plt.yticks([i for i in range(0,5500,500)])
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.title('The length of news data')
plt.show()

label_arr=[0]*num_classes #
label_dict=dict() #
for i in y_train: #
    label_arr[i]+=1 #
for x,y in enumerate(label_arr): #
  label_dict[x]=y #
print("각 레이블에 대한 빈도수") #
print(label_dict)

fig, axe = plt.subplots(ncols=1)
plt.title('The labels of news',fontdict= {'fontsize': 20})
for i in range(46):
    plt.text(i,label_arr[i],label_arr[i],horizontalalignment='center',verticalalignment='bottom')
plt.xlabel('label',fontdict= {'fontsize': 16})
plt.ylabel('number of label',fontdict= {'fontsize': 16})
fig.set_size_inches(20,7)
sns.countplot(y_train)

# # loss 그래프
# epochs1 = range(1, len(history.history['acc']) + 1)
# plt.figure(figsize=(20,8))
# plt.plot(epochs1, history.history['loss'])
# plt.plot(epochs1, history.history['val_loss'])
# plt.title('Model loss',fontdict= {'fontsize': 16})
# plt.xticks([i for i in range(0,len(history.history['acc'])+1,1)])
# plt.ylabel('loss',fontdict= {'fontsize': 14})
# plt.xlabel('epoch',fontdict= {'fontsize': 14})
# plt.legend(['train', 'test'], loc='upper left')
# plt.grid(True)
# plt.show()
#
#
# # acc 그래프
# plt.figure(figsize=(20,8))
# plt.plot(epochs1, history.history['acc'])
# plt.plot(epochs1, history.history['val_acc'])
# plt.title('Model accuracy',fontdict= {'fontsize': 16})
# plt.xticks([i for i in range(0,len(history.history['acc'])+1,1)])
# plt.ylabel('acc',fontdict= {'fontsize': 14})
# plt.xlabel('epoch',fontdict= {'fontsize': 14})
# plt.legend(['train', 'test'], loc='upper left')
# plt.grid(True)
# plt.show()



# -------------------------------------------------------------------



# 2. Transformer를 사용한 reuter 뉴스 분류
# 코드 작성자 : 이경준

# 참고 사이트 링크
# https://www.tensorflow.org/api_docs/python/tf/keras/
# https://wikidocs.net/103802
# https://keras.io/examples/nlp/text_classification_with_transformer/

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential

# 텍스트 분류를 하기 위해서 트랜스포머의 인코딩부분만을 사용한다.
# 인코딩의 구조 : 단어 임베딩 -> positional encoding -> 멀티헤드 어텐션 -> 포지션와이즈 FFNN
# 이후 softmax함수를 추가함


# 1) 단어 임베딩
# 케라스 라우터 데이터에서 이미 정수임베딩되어 있다.
# 데이터 전처리
vocab_size = 40000  # 빈도수 상위 2만개의 단어만 사용
max_len = 200  # 문장의 최대 길이
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.reuters.load_data(num_words=None, test_split=0.2)

print('훈련용 리뷰 개수 : {}'.format(len(X_train)))
print('테스트용 리뷰 개수 : {}'.format(len(X_test)))

X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_len)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_len)


# 2) positional encoding
# 위치 정보를 추가하여 준다.
# 포지션 임베딩
class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_len, vocab_size, embedding_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.pos_emb = tf.keras.layers.Embedding(max_len, embedding_dim)

    def call(self, x):
        max_len = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=max_len, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


# 3) 멀티헤드 어텐션

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim # d_model
        self.num_heads = num_heads

        assert embedding_dim % self.num_heads == 0

        self.projection_dim = embedding_dim // num_heads
        self.query_dense = tf.keras.layers.Dense(embedding_dim)
        self.key_dense = tf.keras.layers.Dense(embedding_dim)
        self.value_dense = tf.keras.layers.Dense(embedding_dim)
        self.dense = tf.keras.layers.Dense(embedding_dim)

    def scaled_dot_product_attention(self, query, key, value):
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        depth = tf.cast(tf.shape(key)[-1], tf.float32)
        logits = matmul_qk / tf.math.sqrt(depth)
        attention_weights = tf.nn.softmax(logits, axis=-1)
        output = tf.matmul(attention_weights, value)
        return output, attention_weights

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]

        # (batch_size, seq_len, embedding_dim)
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # (batch_size, num_heads, seq_len, projection_dim)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention, _ = self.scaled_dot_product_attention(query, key, value)
        # (batch_size, seq_len, num_heads, projection_dim)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # (batch_size, seq_len, embedding_dim)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.embedding_dim))
        outputs = self.dense(concat_attention)
        return outputs


# 4) 포지션와이즈 FFNN
def point_wise_feed_forward_network(embedding_dim, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(embedding_dim)  # (batch_size, seq_len, d_model)
    ])


# 인코더
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads, dff, rate=0.1):
        super(TransformerBlock, self).__init__()
        # self.att = tf.keras.layers.MultiHeadAttention(embedding_dim, num_heads)
        self.att = MultiHeadAttention(embedding_dim, num_heads)
        self.ffn = point_wise_feed_forward_network(embedding_dim, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        # attn_output = self.att(inputs, inputs) # 첫번째 서브층 : 멀티 헤드 어텐션
        attn_output = self.att(inputs) # 첫번째 서브층 : 멀티 헤드 어텐션

        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output) # Add & Norm
        ffn_output = self.ffn(out1) # 두번째 서브층 : 포지션 와이즈 피드 포워드 신경망
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output) # Add & Norm



# optimizer (learning rate가 변화하도록 설정)
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)



embedding_dim = 32  # 각 단어의 임베딩 벡터의 차원
num_heads = 2  # 어텐션 헤드의 수
dff = 32  # 포지션 와이즈 피드 포워드 신경망의 은닉층의 크기
num_classes = 46  # 분류할 클래스의 수

learning_rate = CustomSchedule(embedding_dim)

#optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
optimizer = tf.keras.optimizers.RMSprop(learning_rate, rho=0.9, momentum=0.0, epsilon=1e-07)

model = Sequential()
model.add(TokenAndPositionEmbedding(max_len, vocab_size, embedding_dim))
model.add(TransformerBlock(embedding_dim, num_heads, dff))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(20, activation="relu", kernel_regularizer=tf.keras.regularizers.L2(0.01)))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))


model.compile(optimizer=optimizer, loss = "sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(X_train, y_train, batch_size=32, epochs=15, validation_data=(X_test, y_test))
print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))

epochs1 = range(1, len(history.history['accuracy']) + 1)
plt.figure(figsize=(20,8))
plt.plot(epochs1, history.history['loss'])
plt.plot(epochs1, history.history['val_loss'])
plt.title('Model loss',fontdict= {'fontsize': 16})
plt.xticks([i for i in range(0,len(history.history['accuracy'])+1,1)])
plt.ylabel('loss',fontdict= {'fontsize': 14})
plt.xlabel('epoch',fontdict= {'fontsize': 14})
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)
plt.show()

epochs1 = range(1, len(history.history['accuracy']) + 1)
plt.figure(figsize=(20,8))
plt.plot(epochs1, history.history['accuracy'])
plt.plot(epochs1, history.history['val_accuracy'])
plt.title('Model accuracy',fontdict= {'fontsize': 16})
plt.xticks([i for i in range(0,len(history.history['accuracy'])+1,1)])
plt.ylabel('acc',fontdict= {'fontsize': 14})
plt.xlabel('epoch',fontdict= {'fontsize': 14})
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)
plt.show()


# -------------------------------------------------------------------


# 3. CNN을 사용한 reuter 뉴스 분류
# 코드 작성자 : 이혜인

# 1) CNN _ 커널 수 256 & 커널 사이즈 3

#패키지 import
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt

#데이터 전처리
vocab_size = 10000
(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=vocab_size, test_split=0.2)

print(X_train[:5])

max_len = 500
X_train = pad_sequences(X_train, maxlen = max_len)
X_test = pad_sequences(X_test, maxlen = max_len)

print('X_train의 크기(shape) :',X_train.shape)
print('X_test의 크기(shape) :',X_test.shape)

print('훈련용 뉴스 기사 : {}'.format(len(X_train)))
print('테스트용 뉴스 기사 : {}'.format(len(X_test)))
num_classes = len(set(y_train))
print('카테고리 : {}'.format(num_classes))

#1D CNN 모델 설계
embedding_dim = 256
dropout_ratio = 0.3
num_filters = 256
kernel_size = 3
hidden_units = 128

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(Dropout(dropout_ratio))
model.add(Conv1D(num_filters, kernel_size, padding='valid', activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_units, activation='relu'))
model.add(Dropout(dropout_ratio))
model.add(Dense(1, activation='sigmoid'))

model.summary()

es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 3)
mc = ModelCheckpoint('best_model.h5', monitor = 'val_acc', mode = 'max', verbose = 1, save_best_only = True)

model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['acc'])
history = model.fit(X_train, y_train, epochs = 15, validation_data = (X_test, y_test), callbacks=[es, mc])

loaded_model = load_model('best_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

epochs1 = range(1, len(history.history['acc']) + 1)
plt.figure(figsize=(20,8))
plt.plot(epochs1, history.history['loss'])
plt.plot(epochs1, history.history['val_loss'])
plt.title('Model loss',fontdict= {'fontsize': 16})
plt.xticks([i for i in range(0,len(history.history['acc'])+1,1)])
plt.ylabel('loss',fontdict= {'fontsize': 14})
plt.xlabel('epoch',fontdict= {'fontsize': 14})
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)
plt.show()

epochs1 = range(1, len(history.history['acc']) + 1)
plt.figure(figsize=(20,8))
plt.plot(epochs1, history.history['acc'])
plt.plot(epochs1, history.history['val_acc'])
plt.title('Model accuracy',fontdict= {'fontsize': 16})
plt.xticks([i for i in range(0,len(history.history['acc'])+1,1)])
plt.ylabel('loss',fontdict= {'fontsize': 14})
plt.xlabel('epoch',fontdict= {'fontsize': 14})
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)
plt.show()

# 2) CNN _ 커널 사이즈 256 & 커널 수 5
#1D CNN 모델 설계
embedding_dim = 256
dropout_ratio = 0.3
num_filters = 256
kernel_size = 5
hidden_units = 128

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(Dropout(dropout_ratio))
model.add(Conv1D(num_filters, kernel_size, padding='valid', activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_units, activation='relu'))
model.add(Dropout(dropout_ratio))
model.add(Dense(1, activation='sigmoid'))

# 동일한 부분은 생략
# 3) MultiCNN

#패키지 import
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense, Input, Flatten, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt

#데이터 전처리
vocab_size = 10000
(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=vocab_size, test_split=0.2)

print(X_train[:5])

max_len = 500
X_train = pad_sequences(X_train, maxlen = max_len)
X_test = pad_sequences(X_test, maxlen = max_len)

print('X_train의 크기(shape) :',X_train.shape)
print('X_test의 크기(shape) :',X_test.shape)

#모델 설계
embedding_dim = 128
dropout_ratio = (0.5, 0.8)
num_filters = 128
hidden_units = 128

model_input = Input(shape = (max_len,))
z = Embedding(vocab_size, embedding_dim, input_length = max_len, name="embedding")(model_input)
z = Dropout(dropout_ratio[0])(z)

conv_blocks = []

for sz in [3, 4, 5]:
    conv = Conv1D(filters = num_filters,
                         kernel_size = sz,
                         padding = "valid",
                         activation = "relu",
                         strides = 1)(z)
    conv = GlobalMaxPooling1D()(conv)
    conv_blocks.append(conv)

z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
z = Dropout(dropout_ratio[1])(z)
z = Dense(hidden_units, activation="relu")(z)
model_output = Dense(1, activation="sigmoid")(z)

model = Model(model_input, model_output)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])

es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 4)
mc = ModelCheckpoint('best_model.h5', monitor = 'val_acc', mode = 'max', verbose = 1, save_best_only = True)

history = model.fit(X_train, y_train, batch_size = 64, epochs=15, validation_split = 0.2, verbose=2, callbacks=[es, mc])

loaded_model = load_model('best_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

epochs1 = range(1, len(history.history['acc']) + 1)
plt.figure(figsize=(20,8))
plt.plot(epochs1, history.history['loss'])
plt.plot(epochs1, history.history['val_loss'])
plt.title('Model loss',fontdict= {'fontsize': 16})
plt.xticks([i for i in range(0,len(history.history['acc'])+1,1)])
plt.ylabel('loss',fontdict= {'fontsize': 14})
plt.xlabel('epoch',fontdict= {'fontsize': 14})
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)
plt.show()

epochs1 = range(1, len(history.history['acc']) + 1)
plt.figure(figsize=(20,8))
plt.plot(epochs1, history.history['acc'])
plt.plot(epochs1, history.history['val_acc'])
plt.title('Model accuracy',fontdict= {'fontsize': 16})
plt.xticks([i for i in range(0,len(history.history['acc'])+1,1)])
plt.ylabel('loss',fontdict= {'fontsize': 14})
plt.xlabel('epoch',fontdict= {'fontsize': 14})
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)
plt.show()

# 4) CNN + LSTM

#패키지 import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils
seed = 0
np.random.seed(seed)
tf.compat.v1.set_random_seed(3)

(X_train, Y_train), (X_test, Y_test) = reuters.load_data(num_words=1500, test_split=0.2)

category = np.max(Y_train) + 1
print(category, '카테고리')
print(len(X_train), '학습용 뉴스 기사')
print(len(X_test), '테스트용 뉴스 기사')
print(X_train[0])
print(Y_train[0])

print('뉴스 기사의 최대 길이 :{}'.format(max(len(l) for l in X_train)))
print('뉴스 기사의 평균 길이 :{}'.format(sum(map(len, X_train))/len(X_train)))

plt.hist([len(s) for s in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

x_train = sequence.pad_sequences(X_train, maxlen=500)
x_test = sequence.pad_sequences(X_test, maxlen=500)
y_train = np_utils.to_categorical(Y_train)
y_test = np_utils.to_categorical(Y_test)

s_len = [len(s) for s in X_train]
print(sum([int(i<=500) for i in s_len]))
print(sum([int(i<=500) for i in s_len])/len(X_train))
print(np.shape(x_train))

#모델 설계
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=30000, output_dim=200, input_length=500),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv1D(64, 5, padding='valid', activation='relu', strides=1),
    tf.keras.layers.MaxPooling1D(pool_size=3),
    tf.keras.layers.LSTM(units=46),
    tf.keras.layers.Dense(46, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['acc'])

model.summary()

history = model.fit(x_train, y_train, epochs=15, batch_size=32, validation_data=(x_test, y_test))

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['acc'], 'g-', label='acc')
plt.plot(history.history['val_acc'], 'k--', label='val_acc')
plt.xlabel('Epoch')
plt.legend()

plt.show()


# -------------------------------------------------------------------


# 4. LSTM을 사용한 reuter 뉴스 분류
# 코드 작성자 : 허상진

# https://wikidocs.net/22933
# https://keras.io/ko/losses/

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import reuters
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.optimizer_v1 import RMSprop
from tensorflow.keras.optimizers import RMSprop

(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=10000, test_split=0.2)

print('훈련용 뉴스 기사 : {}'.format(len(X_train)))
print('테스트용 뉴스 기사 : {}'.format(len(X_test)))
num_classes = len(set(y_train))
print('카테고리 : {}'.format(num_classes))


print('뉴스 기사의 최대 길이 :{}'.format(max(len(l) for l in X_train)))
print('뉴스 기사의 평균 길이 :{}'.format(sum(map(len, X_train))/len(X_train)))

'''
plt.hist([len(s) for s in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()
'''

(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=10000, test_split=0.2)

max_len = 200
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

vocab_size = 1000
embedding_dim = 128
hidden_units = 128
num_classes = 46

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(LSTM(hidden_units))
model.add(Dense(num_classes, activation='sigmoid'))
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.001, rho=0.9, epsilon=None, decay=0.0), metrics=['acc'])
history = model.fit(X_train, y_train, batch_size=128, epochs=30, callbacks=[mc,es], validation_data=(X_test, y_test))
loaded_model = load_model('best_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))


epochs1 = range(1, len(history.history['acc']) + 1)
plt.figure(figsize=(20,8))
plt.plot(epochs1, history.history['loss'])
plt.plot(epochs1, history.history['val_loss'])
plt.title('Model loss',fontdict= {'fontsize': 16})
plt.xticks([i for i in range(0,len(history.history['acc'])+1,1)])
plt.ylabel('loss',fontdict= {'fontsize': 14})
plt.xlabel('epoch',fontdict= {'fontsize': 14})
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)
plt.show()

epochs1 = range(1, len(history.history['acc']) + 1)
plt.figure(figsize=(20,8))
plt.plot(epochs1, history.history['acc'])
plt.plot(epochs1, history.history['val_acc'])
plt.title('Model accuracy',fontdict= {'fontsize': 16})
plt.xticks([i for i in range(0,len(history.history['acc'])+1,1)])
plt.ylabel('loss',fontdict= {'fontsize': 14})
plt.xlabel('epoch',fontdict= {'fontsize': 14})
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)
plt.show()