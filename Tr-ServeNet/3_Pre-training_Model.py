import pickle
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import csv
import h5py
import pandas as pd
import numpy as np
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,4,6,7"
# from sklearn.model_selection import train_test_split
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])  # TPU detection
    BUCKET = 'servenet'  # @param {type:"string"}
    assert BUCKET, 'Must specify an existing GCS bucket name'
    OUTPUT_DIR = 'gs://{}/tfhub-modules-cache'.format(BUCKET)
    tf.io.gfile.makedirs(OUTPUT_DIR)
    print('***** Model output directory: {} *****'.format(OUTPUT_DIR))
    os.environ["TFHUB_CACHE_DIR"] = OUTPUT_DIR

except:
    tpu = None
    gpus = tf.config.experimental.list_logical_devices("GPU")
    os.environ["TFHUB_CACHE_DIR"] = "Model"

# Select appropriate distribution strategy
if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(
        tpu)  # Going back and forth between TPU and host is expensive. Better to run 128 batches on the TPU before reporting back.
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
elif len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
    print('Running on multiple GPUs ', [gpu.name for gpu in gpus])
elif len(gpus) == 1:
    strategy = tf.distribute.get_strategy()  # default strategy that works on CPU and single GPU
    print('Running on single GPU ', gpus[0].name)
else:
    strategy = tf.distribute.get_strategy()  # default strategy that works on CPU and single GPU
    print('Running on CPU')
print("Number of accelerators: ", strategy.num_replicas_in_sync)

from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras import metrics
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Activation, Conv2D, Reshape, Average, Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Concatenate, Lambda

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.initializers import Orthogonal

maxName = 10
maxDes = None

bert_path = "https://hub.tensorflow.google.cn/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"
hiddenSize = 768

f_train = open('Data/BERT-AppDatasetWithNameMiniBatch-TrainData-26-200d.pickle','rb')
f_test = open('Data/BERT-AppDatasetWithNameMiniBatch-TestData-26-200d.pickle', 'rb')
traindata = pickle.load(f_train)
testdata = pickle.load(f_test)
f_train.close()
f_test.close()
tranning_steps_per_epoch = len(traindata)
validation_steps = len(testdata)


def train_generator():
    while True:
        for d in traindata:
            x_train = d[0]
            y_train = d[1]
            yield x_train, y_train


def test_generator():
    while True:
        for d in testdata:
            x_test = d[0]
            y_test = d[1]
            yield x_test, y_test


class WeightedLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(WeightedLayer, self).__init__(**kwargs)

        self.w1 = self.add_weight(name='w1', shape=(1), initializer="ones", dtype=tf.float32, trainable=True)
        self.w2 = self.add_weight(name='w2', shape=(1), initializer="ones", dtype=tf.float32, trainable=True)

    def call(self, inputs1, inputs2):
        return inputs1 * self.w1 + inputs2 * self.w2

    def get_config(self):
        config = super(WeightedLayer, self).get_config()
        return config


def ServeNet():
    """
    Function creating the ServeNet model

    Arguments:
    input_shape -- shape of the input, usually (max_len, max_len_name)

    Returns:
    model -- a model instance in Keras
    """

    # INPUT - Service Name
    in_name_id = Input(shape=(maxName,), dtype=tf.int32, name="input_word_name_ids")
    in_name_mask = Input(shape=(maxName,), dtype=tf.int32, name="input_name_masks")
    in_name_segment = Input(shape=(maxName,), dtype=tf.int32, name="segment_name_ids")

    bert_name_inputs = [in_name_id, in_name_mask, in_name_segment]

    # BERT for Name
    bert_name_layer = hub.KerasLayer(bert_path, trainable=True, name="bert_name")
    pooled_output, _ = bert_name_layer(bert_name_inputs)

    # name_embeddings = Reshape((maxLen, hiddenSize, 1))(pooled_output)

    # Feature for Name
    name_features = Dense(1024, activation='tanh', name="name_feature",
                          kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(pooled_output)
    name_features = Dropout(0.1)(name_features)

    # name_features = pooled_output

    # INPUT - Service Description
    in_id = Input(shape=(maxDes,), dtype=tf.int32, name="input_word_ids")
    in_mask = Input(shape=(maxDes,), dtype=tf.int32, name="input_masks")
    in_segment = Input(shape=(maxDes,), dtype=tf.int32, name="segment_ids")
    bert_description_inputs = [in_id, in_mask, in_segment]

    # BERT for Description
    bert_layer = hub.KerasLayer(bert_path, trainable=True, name="bert_description")
    _, sequence_output = bert_layer(bert_description_inputs)

    embeddings = Reshape((-1, hiddenSize, 1))(sequence_output)

    # CNN
    features1 = Conv2D(32, kernel_size=(3, 3), padding='same')(embeddings)
    features1 = Dropout(0.1)(features1)
    features2 = Conv2D(1, kernel_size=(1, 1), padding='same')(features1)
    # features2 = Dropout(0.4)(features2)

    features = Reshape((-1, hiddenSize))(features2)

    # features = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=8)(sequence_output,sequence_output)

    # LSTM
    description_features = Bidirectional(LSTM(512, return_sequences=False, name="description_feature"))(features)
    description_features = Dropout(0.1)(description_features)

    # Merge Features
    all_features = WeightedLayer()(name_features, description_features)
    # all_features = Concatenate(name="allfeatures")([name_features, description_features])

    # TASK
    # X = Dense(200, activation='tanh', kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(X)
    # X = Dropout(0.1)(X)
    output = Dense(50, activation='softmax', kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))(
        all_features)

    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=[bert_description_inputs, bert_name_inputs], outputs=output)

    ### END CODE HERE ###

    return model


SavePrefix = 'ServeNet-BERT-ServiceName-DataGenerator-More-26'
checkpointer = [ModelCheckpoint(
    filepath='Data/ServeNet-ServeNet-BERT-ServiceName-DataGenerator-More-26-top5.hdf5',
    monitor='val_top_k_categorical_accuracy', verbose=1, save_best_only=True),
                ModelCheckpoint(
                    filepath='Data/ServeNet-ServeNet-BERT-ServiceName-DataGenerator-More-26-top1.hdf5',
                    monitor='val_categorical_accuracy', verbose=1, save_best_only=True)]
sgd = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.8, nesterov=True)
model = ServeNet()
model.compile(loss='categorical_crossentropy', optimizer="sgd",
              metrics=[metrics.top_k_categorical_accuracy, metrics.categorical_accuracy])

history = model.fit(train_generator(), steps_per_epoch=tranning_steps_per_epoch,
                    validation_data=test_generator(), validation_steps=validation_steps,
                    epochs=30, verbose=1, callbacks=[checkpointer])

print("Training set:")
loss_train, top5error_train, top1error_train = model.evaluate(train_generator(), steps=tranning_steps_per_epoch)
print("Top5 Training accuracy = ", top5error_train)
print("Top1 Training accuracy = ", top1error_train)

print('Test set:')
loss_test, top5error_test, top1error_test = model.evaluate(test_generator(), steps=validation_steps)
print("Top5 Test accuracy = ", top5error_test)
print("Top1 Test accuracy = ", top1error_test)
