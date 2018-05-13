import json
import numpy as np
import itertools
from operator import itemgetter
from feature_extraction import string_match_features, get_distance_features


EMBEDDING_DIM = 300
NEGATIVE = 0.7

global groups
global sorted_groups

def fillin_global_variables():
    global groups
    global sorted_groups

    with open('./data/groups_in_chains.json', 'r') as fin:
        groups = json.load(fin)

    with open('./data/sorted_groups.json') as fin:
        sorted_groups = json.load(fin)


def create_pair_dataset():
    """
    Create dataset of groups pairs: (group1_id, group2_id, label) --> dataset_pair.txt.
    NEGATIVE constant is the ratio of negatives out of positives.
    :return:
    """

    fillin_global_variables()

    # with open('./data/dataset_pair_text_lem.txt', 'w') as fout:
    with open('./data/dataset_pair_with_features.txt', 'w') as fout:
        for doc_id, chains in groups.items():

            print('{} document is processed...'.format(doc_id))
            # positive pairs
            num_of_positive = 0
            for chn in chains:
                for pair in itertools.combinations(chn['groups'], 2):
                    # string match amd distance features
                    pair_features = np.array_str(get_pos_pair_features(pair[0], pair[1], doc_id))
                    fout.write('{} {} {} {}\n'.format(pair[0]['group_id'], pair[1]['group_id'], 1, pair_features[1:-1]))

                    # content1 = [itm['lem'] for itm in pair[0]['items']]
                    # content2 = [itm['lem'] for itm in pair[1]['items']]
                    # content1 = ' '.join(content1)
                    # content2 = ' '.join(content2)
                    # fout.write('{} {} {}\n'.format(content1, content2, 1))

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
                        pair_features = np.array_str(get_neg_pair_features(pair[0], pair[1], doc_id))
                        fout.write('{} {} {} {}\n'.format(pair[0][0], pair[1][0], 0, pair_features[1:-1]))

                        # fout.write('{} {} {}\n'.format(pair[0][3], pair[1][3], 0))

                        num_of_negative += 1
                else:
                    break

            print('{} negative pairs created.'.format(num_of_negative))


def get_pos_pair_features(group1, group2, doc_id):
    # np.array -> [exact match, head match and part match]
    str_features = string_match_features(group1, group2)

    # np.array -> [interven groups]
    idx1 = next((i for i, grp in enumerate(sorted_groups[doc_id]) if grp[0] == group1['group_id']), 0)
    idx2 = next((i for i, grp in enumerate(sorted_groups[doc_id]) if grp[0] == group2['group_id']), 0)

    dist_features = get_distance_features(idx1, idx2)

    return np.r_[str_features, dist_features]


def get_neg_pair_features(group1, group2, doc_id):
    # np.array -> [exact match, head match and part match]
    chn1 = next((chn for chn in groups[doc_id] if chn['chain_id'] == group1[1]), None)
    chn2 = next((chn for chn in groups[doc_id] if chn['chain_id'] == group2[1]), None)

    grp1 = next((grp for grp in chn1['groups'] if grp['group_id'] == group1[0]), None)
    grp2 = next((grp for grp in chn2['groups'] if grp['group_id'] == group2[0]), None)

    str_features = string_match_features(grp1, grp2)

    # np.array -> [interven groups]
    idx1 = sorted_groups[doc_id].index(group1)
    idx2 = sorted_groups[doc_id].index(group2)

    dist_features = get_distance_features(idx1, idx2)

    return np.r_[str_features, dist_features]


def sord_pair_dataset_by_mention():
    dataset = []

    with open('./data/dataset_pair.txt', 'r') as fin:
        for line in fin:
            entry = [int(val) for val in line.split()]
            dataset.append(entry)

    dataset = sorted(dataset, key=itemgetter(1))

    with open('./data/dataset_pair_sorted.txt', 'w') as fout:
        for entry in dataset:
            fout.write('{} {} {}\n'.format(entry[0], entry[1], entry[2]))

    return dataset


def main():

    create_pair_dataset()
    # sord_pair_dataset_by_mention()


if __name__ == '__main__':
    main()
