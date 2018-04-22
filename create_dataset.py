from gensim.models.fasttext import FastText
import numpy as np
import json
import itertools


EMBEDDING_DIM = 300
NEGATIVE = 0.7

# groups = {}
# sorted_groups = {}
# dataset_pair = np.array([])
# dataset_single = np.array([])
# embedding_matrix = []
# word_index = {}
#
#
# def fillin_global_variables():
#     global groups
#     global sorted_groups
#     global word_index
#     global embedding_matrix
#
#     with open('./data/groups_in_chains.json', 'r') as fin:
#         groups = json.load(fin)
#
#     with open('./data/sorted_groups.json') as fin:
#         sorted_groups = json.load(fin)
#
#     if not word_index:
#         word_index = {}
#         with open('./data/word_index.txt', 'r') as fin:
#             for line in fin:
#                 line = line.split()
#                 word_index[line[0]] = int(line[1])
#
#     if not embedding_matrix:
#         embedding_matrix = np.load('./data/embedding_matrix.npy')
#
#
# def add_entry_to_dataset(group, antecedent=None, label=None):
#     global dataset_single
#     global dataset_pair
#
#     head = next((itm['lem'] for itm in group['items'] if 'head' in itm), group['items'][0]['lem'])
#     head_vec = embedding_matrix[word_index[head]]
#
#     # antecedent is NA
#     if not antecedent:
#         features = []
#
#         if dataset_single.size == 0:
#             dataset_single = np.concatenate((head_vec, features, [1]))
#         else:
#             dataset_single = np.vstack((dataset_single, np.concatenate((head_vec, features, [1]))))
#
#     else:
#         anteced_head = next((itm['lem'] for itm in antecedent['items'] if 'head' in itm), antecedent['items'][0]['lem'])
#         anteced_head_vec = embedding_matrix[word_index[anteced_head]]
#
#         if dataset_pair.size == 0:
#             dataset_pair = np.concatenate((anteced_head_vec, head_vec, label))
#         else:
#             dataset_pair = np.vstack((dataset_pair, np.concatenate((anteced_head_vec, head_vec, label))))
#
#
# def make_negative_dataset(doc_id, max_negative):
#
#     num_of_negative = 0
#     doc_groups = sorted_groups[doc_id]
#
#     for pair in itertools.combinations(doc_groups, 2):
#         # if both groups belong to 1 chain
#         if pair[0][1] == pair[1][1]:
#             continue
#         # if max_negative limit is not over
#         elif num_of_negative < max_negative:
#             chn = next((chn for chn in groups[doc_id] if chn['chain_id'] == pair[0][1]), None)
#             antecedent = next((grp for grp in chn['groups'] if grp['group_id'] == pair[0][0]), None)
#
#             chn = next((chn for chn in groups[doc_id] if chn['chain_id'] == pair[1][1]), None)
#             group = next((grp for grp in chn['groups'] if grp['group_id'] == pair[1][0]), None)
#
#             add_entry_to_dataset(group, antecedent, [0])
#             num_of_negative += 1
#         else:
#             return
#
#
# def create_dataset_old():
#
#     fillin_global_variables()
#
#     for doc_id, chains in groups.items():
#         num_of_positive = 0
#         for chn in chains:
#             # mentions within chain - with label '1'
#             for pair in itertools.combinations(chn['groups'], 2):
#                 add_entry_to_dataset(pair[1], pair[0], [1])
#                 num_of_positive += 1
#
#         # add negative pairs to dataset
#         make_negative_dataset(doc_id, round(NEGATIVE * num_of_positive))
#
#     np.save('/output/dataset_pair.npy', dataset_pair)
#     # np.save('/output/dataset_single.npy', dataset_single)

def create_pair_dataset():
    """
    Create dataset of groups pairs: (group1_id, group2_id, label) --> dataset_pair.txt.
    NEGATIVE constant is the ratio of negatives out of positives.
    :return:
    """
    with open('./data/groups_in_chains.json', 'r') as fin:
        groups = json.load(fin)

    with open('./data/sorted_groups.json') as fin:
        sorted_groups = json.load(fin)

    with open('./data/dataset_pair.txt', 'w') as fout:
        for doc_id, chains in groups.items():

            print('{} document is processed...'.format(doc_id))
            # positive pairs
            num_of_positive = 0
            for chn in chains:
                for pair in itertools.combinations(chn['groups'], 2):
                    labelled_pair = (pair[0]['group_id'], pair[1]['group_id'], 1)
                    fout.write('{} {} {}\n'.format(labelled_pair[0], labelled_pair[1], labelled_pair[2]))
                    num_of_positive += 1

            print('{} positive pairs created.'.format(num_of_positive))

            # negative pairs
            max_negative = round(num_of_positive * NEGATIVE)
            num_of_negative = 0
            doc_groups = sorted_groups[doc_id]

            for pair in itertools.combinations(doc_groups, 2):
                # if max_negative limit is not over
                if num_of_negative < max_negative:
                    # if both groups belong to 1 chain
                    if pair[0][1] == pair[1][1]:
                        continue
                    else:
                        labelled_pair = (pair[0][0], pair[1][0], 0)
                        fout.write('{} {} {}\n'.format(labelled_pair[0], labelled_pair[1], labelled_pair[2]))
                        num_of_negative += 1
                else:
                    break

            print('{} negative pairs created.'.format(num_of_negative))


def main():

    create_pair_dataset()


if __name__ == '__main__':
    main()
