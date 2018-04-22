from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import numpy as np


dataset_features = np.array([])
group_index = {}


class Model:
    def __init__(self, filepath=None):
        if not filepath:
            self.model = Sequential()
        else:
            self.model = load_model(filepath)

    def set(self):
        self.model.add(Dense(units=256, init='uniform', activation='relu', input_dim=1200))
        self.model.add(Dense(units=128, init='uniform', activation='relu'))
        self.model.add(Dense(units=64, init='uniform', activation='relu'))
        self.model.add(Dense(units=32, init='uniform'))
        self.model.add(Dense(units=1, activation='sigmoid'))

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, x_train, y_train, batch_size, epochs):
        self.model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs)
        self.model.save('/output/CRModel.h5')
        self.model.save_weights('/output/CRModel_weights.h5')
        TensorBoard(log_dir='/output/logs', histogram_freq=2, batch_size=100, write_graph=False,
                                    write_grads=True, write_images=False, embeddings_freq=0,
                                    embeddings_layer_names=None, embeddings_metadata=None)

    # def train_iterative(self, steps=1000, epochs=5):
    #     self.model.fit_generator(generate_group_pairs(), steps_per_epoch=steps, epochs=epochs)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def estimate_metrics(self, x_test, y_test):
        labels_pred = self.predict(x_test)
        labels_pred = (labels_pred > 0.5)

        print('Labels predicted: ', str(labels_pred))

        accuracy = metrics.accuracy_score(y_test, labels_pred)
        metric = metrics.precision_recall_fscore_support(y_test, labels_pred)

        print('Accuracy is ' + str(accuracy))
        print('Metrics (precision, recall, f score, support) is ' + str(metric))


def fillin_global_variables():
    global dataset_features
    global group_index

    # from Floyd /dataset/
    dataset_features = np.load('/dataset/dataset_mention_features.npy')

    with open('./data/group_index.txt', 'r') as fin:
        for line in fin:
            line = line.split()
            group_index[int(line[0])] = int(line[1])


def get_dataset_entry(line):
    values = [int(val) for val in line.split()]
    id1 = values[0]
    id2 = values[1]
    label = [values[2]]

    entry = np.concatenate((dataset_features[group_index[id1]], dataset_features[group_index[id2]], label))

    # Add pair features later
    # ...

    return entry


# def generate_group_pairs():
#     while True:
#         with open('./data/dataset_pair.txt', 'r') as fin:
#             for line in fin:
#                 # create Numpy arrays of input data
#                 # and labels, from each line in the file
#                 data, label = process_line(line)
#                 yield (data, label)


def generate_full_dataset():

    dataset = []

    fillin_global_variables()

    # generate dataset for training and testing
    with open('./data/dataset_pair.txt', 'r') as fin:
        for line in fin:
            entry = get_dataset_entry(line)
            dataset.append(entry)

    dataset = np.array(dataset)

    return dataset


def preprocess_dataset(dataset):
    """
    Load, shuffle and split dataset into train and test ones.
    :return:
    """
    # dataset_pair = np.load('./data/dataset/dataset_pair.npy')
    # np.random.shuffle(dataset_pair)
    #
    # data = dataset_pair[:, :-1]
    # labels = dataset_pair[:, -1]

    np.random.shuffle(dataset)

    data = dataset[:, :-1]
    labels = dataset[:, -1]

    return train_test_split(data, labels, test_size=0.2)


def pipeline():

    dataset = generate_full_dataset()
    print('Dataset is generated.')
    data_train, data_test, labels_train, labels_test = preprocess_dataset(dataset)
    print('Dataset is preprocessed.')

    # del dataset

    batch_size = 1000
    epochs = 10

    model = Model()
    model.set()
    print('Model is compiled. Start training.')
    model.train(data_train, labels_train, batch_size, epochs)
    print('Training is finished.')

    print('ESTIMATE MODEL')
    model.estimate_metrics(data_test, labels_test)


def play_with_model(data_test, labels_test):
    # data_train, data_test, labels_train, labels_test = preprocess_dataset()
    # print('Dataset is preprocessed.')

    model = Model('./models/CRModel.h5')
    print('Model is downloaded.')

    model.estimate_metrics(data_test, labels_test)


def load_vocabs():
    word_index = {}
    with open('./data/word_index_lemma.txt', 'r') as fin:
        for line in fin:
            line = line.split()
            word_index[line[0]] = int(line[1])

    embedding_matrix = np.load('./data/embedding_matrix_lemma.npy')

    return word_index, embedding_matrix

def main():
    wi, em = load_vocabs()
    m1 = em[wi['мужчина']]
    m2 = em[wi['он']]
    m3 = em[wi['она']]
    pair1 = np.concatenate((m1, m1, m2, m2, [1]))
    pair2 = np.concatenate((m1, m1, m3, m3, [0]))
    pair3 = np.concatenate((m2, m2, m3, m3, [0]))
    dataset = np.vstack((pair1, pair2, pair3))
    data = dataset[:, :-1]
    labels = dataset[:, -1]

    play_with_model(data, labels)

    # pipeline()



if __name__ == '__main__':
    main()