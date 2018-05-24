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

# from tensorflow.python import debug as tf_debug

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

        # if self.mode == 'train':
        #     # dummy = tf.expand_dims(tf.constant([0.]), 0)
        #     # dummy = tf.constant([0.])
        #     model.add(Lambda(lambda x: tf.stack([x, tf.constant([0.], shape=x.get_shape())])))

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

        # Debugging
        # K.set_session(tf_debug.LocalCLIDebugWrapperSession(tf.Session()))

        history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                                 validation_split=0.2, shuffle=False, verbose=2)

        # if self.mode == 'pretrain':
        #     self.model.save('/output/CRModel.h5')
        #     self.model.save_weights('/output/CRModel_weights.h5')
        # else:
        #     self.model.save('/output/CRModel_final.h5')

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
    # mentions_sorted = tf.Print(mentions_sorted, [mentions_sorted], "mentions_sorted:", first_n=1, summarize=100)
    # print('Mentions_sorted: {}'.format(mentions_sorted))

    labels_sorted = K.gather(y_true[:, 0], indices_sorted)
    # labels_sorted = tf.Print(labels_sorted, [labels_sorted], "labels_sorted:", first_n=1, summarize=100)
    # print('Labels_sorted: {}'.format(labels_sorted))

    scores_sorted = K.gather(y_pred[:, 0], indices_sorted)
    # scores_sorted = tf.Print(scores_sorted, [scores_sorted], "scores_sorted:", first_n=1, summarize=100)
    # print('Scores_sorted: {}'.format(scores_sorted))

    return max_margin(labels_sorted, mentions_sorted, scores_sorted)


def max_margin(labels, mentions, scores):
    batch = 100

    def different(start, end):
        indices = tf.range(start, end, delta=1)
        # indices = tf.Print(indices, [tf.gather(mentions, indices)], 'indices: ', first_n=50, summarize=40)
        labels_to_process = tf.gather(labels, indices)
        predictions_to_process = tf.gather(scores, indices)

        return process_antecedents(labels_to_process, predictions_to_process)

    def mention_group_loss(i, prev_mention, i_start):
        condition = tf.equal(tf.gather(mentions, i), prev_mention)
        m_loss = tf.cond(condition, lambda: K.constant(0.), lambda: different(i_start, i))

        return m_loss

    def set_i_start(i, prev_mention, i_start):
        condition = tf.equal(tf.gather(mentions, i), prev_mention)

        return tf.cond(condition, lambda: i_start, lambda: i)

    i0 = tf.constant(0)
    prev_mention0 = tf.gather(mentions, i0)
    i_start0 = tf.constant(0)
    loss0 = tf.constant(0.)

    c = lambda i, prev_mention, i_start, loss: tf.less(i, batch)
    b = lambda i, prev_mention, i_start, loss: [i + 1,
                                                tf.gather(mentions, i),
                                                set_i_start(i, prev_mention, i_start),
                                                loss + mention_group_loss(i, prev_mention, i_start)]

    loop = tf.while_loop(c, b, [i0, prev_mention0, i_start0, loss0], name='while_loop')
    loss = loop[3]

    return loss


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

    n = 10000
    data = dataset[:n, :-2]
    if mode == 'pretrain':
        labels = dataset[:n, -2]
    else:
        labels = dataset[:n, -2:]

    return train_test_split(data, labels, test_size=0.05, shuffle=False)


def pipeline(batch_size, epochs, mode='pretrain', filepath=None):
    # dataset = generate_full_dataset()
    # print('Dataset is generated.')
    dataset = np.load('./data/dataset/full_dataset_no_duplicates.npy')
    print('Dataset is loaded.')

    data_train, data_test, labels_train, labels_test = preprocess_dataset(dataset, mode)
    print('Dataset is preprocessed.')

    model = Model(mode=mode, filepath=filepath)
    model.train(data_train, labels_train, batch_size, epochs)
    #
    # print('Evaluation:')
    # model.estimate_metrics(data_test, labels_test)

    # np.save('/output/data_test.npy', np.c_[data_test, labels_test])


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


def main():
    # Pre-training
    # pipeline(batch_size=100, epochs=20)

    # Training with max-margin loss
    pipeline(batch_size=100, epochs=1, mode='train', filepath='CRModel_weights.h5')

    # evaluate_model()


def check():
    s = tf.constant(0)
    e = tf.constant(3)
    arr = tf.constant([0, 1, 2, 3, 4, 5])

    r = tf.range(s, e, delta=1)
    res = tf.gather(arr, r)

    with tf.Session() as sess:
        print(sess.run(res))


def check_count(lab, pred):
    labels = np.array([int(l) for l in lab.split()])
    predictions = np.array([float(p) for p in pred.split()])

    true_scores = labels * predictions
    highest_true_score = np.max(true_scores)
    penalties = 1.0 * (1 - labels)

    result_arr = penalties * (1 + predictions - highest_true_score)
    result_max = np.max(result_arr)

    print("Result array: {}\nMax loss: {}".format(result_arr, result_max))


if __name__ == '__main__':
    main()
