from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda
from keras.optimizers import RMSprop, Adam
from keras.constraints import maxnorm
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import numpy as np
# import matplotlib.pyplot as plt
import bcubed

# from generate_full_dataset import generate_full_dataset

PROB_THRESHOLD = 0.7


class Model:
    def __init__(self, mode='pretrain', filepath=None):
        self.mode = mode

        if mode == 'test':
            self.model = load_model(filepath, custom_objects={'max_margin_loss': max_margin_loss})
        else:
            self.model = self.set()
            if mode == 'train':
                self.model.load_weights(filepath)
            self.compile()

    def set(self):
        model = Sequential()
        # model.add(Dropout(0.3, input_shape=(1238,))) # makes loss worse
        model.add(Dense(units=500, kernel_initializer='normal', activation='relu', input_dim=1238,
                        kernel_constraint=maxnorm(3)))
        # model.add(Dense(units=1000, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)))
        model.add(Dropout(0.3))
        model.add(Dense(units=250, kernel_initializer='normal', activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(units=100, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)))
        model.add(Dropout(0.3))
        model.add(Dense(units=1, kernel_initializer='normal', activation='sigmoid'))

        if self.mode == 'train':
            dummy = tf.expand_dims(tf.constant([0.]), 0)
            model.add(Lambda(lambda x: tf.stack([x, dummy])))

        return model

    def compile(self):
        optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
        loss = 'binary_crossentropy'

        if self.mode == 'train':
            optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-07, decay=0.0)
            loss = max_margin_loss

        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        print('Model is compiled. Start training.')

    def train(self, x_train, y_train, batch_size, epochs):
        # tensorboard = TensorBoard(log_dir='/output/logs', histogram_freq=2, batch_size=100, write_graph=False,
        #                           write_grads=True, write_images=False, embeddings_freq=0,
        #                           embeddings_layer_names=None, embeddings_metadata=None)
        history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                                 validation_split=0.2, shuffle=True, verbose=2)

        if self.mode == 'pretrain':
            self.model.save('/output/CRModel.h5')
            self.model.save_weights('/output/CRModel_weights.h5')
        else:
            self.model.save('/output/CRModel_final.h5')

    def predict(self, x_test):
        return self.model.predict(x_test)

    # Computes the loss on test data
    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def estimate_metrics(self, x_test, y_test):
        labels_pred = self.predict(x_test)
        labels_pred = (labels_pred > PROB_THRESHOLD)

        print('Labels predicted: ', str(labels_pred))

        accuracy = metrics.accuracy_score(y_test, labels_pred)
        metric = metrics.precision_recall_fscore_support(y_test, labels_pred)

        print('Accuracy is ' + str(accuracy))
        print('Metrics (precision, recall, f score, support) is ' + str(metric))

    def bcubed(self, x_test, y_test):
        ldict = {}
        cdict = {}

        labels_pred = self.predict(x_test)
        labels_pred = (labels_pred > PROB_THRESHOLD)

        for i, label in enumerate(y_test):
            ldict[i] = {int(label)}
            cdict[i] = {int(labels_pred[i])}

        precision = bcubed.precision(cdict, ldict)
        recall = bcubed.recall(cdict, ldict)
        fscore = bcubed.fscore(precision, recall)

        print('B-cubed metric:\nPrecision = {}\nRecall = {}\nF-score = {}'.format(precision, recall, fscore))


# LOSS FUNCTION

def max_margin_loss(y_true, y_pred):

    k = tf.shape(y_true)[0]

    mentions_sorted, indices_sorted = tf.nn.top_k(y_true[:, 1], k, sorted=True)

    labels_sorted = K.gather(y_true[:, 0], indices_sorted)
    scores_sorted = K.gather(y_pred[:, 0], indices_sorted)

    # true_scores = y_true * y_pred
    # highest_true_score = K.max(true_scores)
    # penalties = penalty * (1 - y_true)
    # return K.max(penalties * (1 + y_pred - highest_true_score))

    return max_margin(labels_sorted, mentions_sorted, scores_sorted)


def max_margin(labels, mentions, scores):
    batch = 100
    current_mention = mentions[0]
    labels_to_process = []
    pred_to_process = []
    loss = []

    def same(label, score):
        labels_to_process.append(label)
        pred_to_process.append(score)
        return K.constant(0.)

    def different():
        nonlocal labels_to_process
        nonlocal pred_to_process

        labels = tf.stack(labels_to_process)
        predictions = tf.stack(pred_to_process)

        labels_to_process = []
        pred_to_process = []

        return process_antecedents(labels, predictions)

    labels_arr = tf.unstack(labels, num=batch)
    mentions_arr = tf.unstack(mentions, num=batch)
    scores_arr = tf.unstack(scores, num=batch)


    for i, mention in enumerate(mentions_arr):
        # columns_true = tf.unstack(r)
        # columns_pred = tf.unstack(rows_pred[i])

        label = labels_arr[i]
        score = scores_arr[i]

        loss.append(tf.cond(tf.equal(mention, current_mention), lambda: same(label, score), lambda: different()))
        current_mention = mention

    return K.sum(loss)


def process_antecedents(labels, predictions):
    penalty = 1.0

    true_scores = labels * predictions
    highest_true_score = K.max(true_scores)
    penalties = penalty * (1 - labels)

    return K.max(penalties * (1 + predictions - highest_true_score))

###


def preprocess_dataset(dataset, mode):
    """
    Load, shuffle and split dataset into train and test ones.
    :return:
    """
    # np.random.shuffle(dataset)

    data = dataset[:, :-2]
    if mode == 'pretrain':
        labels = dataset[:, -2]
    else:
        labels = dataset[:, -2:]

    return train_test_split(data, labels, test_size=0.2)


def pipeline(batch_size, epochs, mode='pretrain', filepath=None):
    # dataset = generate_full_dataset()
    # print('Dataset is generated.')
    dataset = np.load('/dataset/full_dataset_no_duplicates.npy')
    print('Dataset is loaded.')

    data_train, data_test, labels_train, labels_test = preprocess_dataset(dataset, mode)
    print('Dataset is preprocessed.')

    # del dataset

    model = Model(mode=mode, filepath=filepath)
    model.train(data_train, labels_train, batch_size, epochs)

    # print('Built-in evaluation:')
    # model.evaluate(data_test, labels_test)
    print('Evaluation:')
    model.estimate_metrics(data_test, labels_test)

    np.save('/output/data_test.npy', np.c_[data_test, labels_test])


def evaluate_model(test_size=None):
    # Download from Floyd datasets through /data/ folder
    model = Model('/dataset/CRModel.h5')
    print('Model is downloaded.')

    dataset = np.load('/dataset/data_test.npy')
    if test_size != None:
        dataset = dataset[:test_size]

    np.random.shuffle(dataset)
    data_test = dataset[:, :-1]
    labels_test = dataset[:, -1]
    print('Dataset for testing is ready.')

    print('ESTIMATE MODEL')
    model.estimate_metrics(data_test, labels_test)
    model.bcubed(data_test, labels_test)


# def plot_history(self, history):
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('model loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'validation'], loc='upper left')
#     plt.show()


def main():
    # Pre-training
    pipeline(batch_size=100, epochs=20)

    # Training with max-margin loss
    # pipeline(batch_size=100, epochs=20, mode='train', filepath='CRModel_weights.h5')

    # evaluate_model()


if __name__ == '__main__':
    main()
