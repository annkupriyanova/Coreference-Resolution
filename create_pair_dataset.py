import json
import itertools


EMBEDDING_DIM = 300
NEGATIVE = 0.7


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
