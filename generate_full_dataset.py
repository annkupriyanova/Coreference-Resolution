import numpy as np
from operator import itemgetter


dataset_features = np.array([])
group_index = {}

def fillin_global_variables():
    global dataset_features
    global group_index

    # from Floyd /dataset/
    dataset_features = np.load('/dataset/dataset_mention_features_basic.npy')

    with open('./data/group_index.txt', 'r') as fin:
        for line in fin:
            line = line.split()
            group_index[int(line[0])] = [int(line[1]), int(line[2])]


def get_dataset_entry(line):
    values = [int(val) for val in line.split()]
    id1 = values[0]
    id2 = values[1]
    label = [values[2]]
    pair_features = np.array(values[3:])

    # entry = np.concatenate((dataset_features[group_index[id1][0]], dataset_features[group_index[id2][0]], label))
    entry = np.concatenate((dataset_features[group_index[id1][0]], dataset_features[group_index[id2][0]], pair_features, label, [id2]))

    return entry


def generate_full_dataset():
    dataset = []

    fillin_global_variables()

    # generate dataset for training and testing
    # with open('./data/dataset_pair_with_features.txt', 'r') as fin:
    with open('./data/dataset_pair_sorted.txt', 'r') as fin:
        for line in fin:
            entry = get_dataset_entry(line)
            dataset.append(entry)

    dataset = np.array(dataset)
    print("Original dataset: {}".format(dataset.shape))

    # delete duplicates
    dataset = delete_conradictions(dataset)
    print("Without duplicates and contradictions: {}".format(dataset.shape))

    dataset = sorted(dataset, key=itemgetter(-1))

    return dataset


# def unique_rows(a):
#     a = np.ascontiguousarray(a)
#     unique_a, unique_i = np.unique(a.view([('', a.dtype)]*a.shape[1]), return_index=True)
#     return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1])), unique_i


def unique_rows(arr):
    return np.unique(arr, return_index=True, axis=0)


def delete_conradictions(dataset):
    labels = dataset[:, -1]

    unique_data, unique_i = unique_rows(dataset[:, :-1])
    unique_labels = labels[unique_i]

    return np.c_[unique_data, unique_labels]


# def main():
#     a = np.array([[1.2, 0.4, 6.7], [0, 1, 3]])
#     a = sorted(a, key=itemgetter(-1))
#
#     print(a)


def save_full_dataset(filepath):

    dataset = generate_full_dataset()

    np.save(filepath, dataset)


if __name__ == '__main__':
    save_full_dataset('/output/full_dataset_no_duplicates.npy')
    # main()