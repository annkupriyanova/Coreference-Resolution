from generate_full_dataset import generate_full_dataset
import numpy as np


def check_duplicates():
    n = 0
    pos = 0
    neg = 0
    pair_counter = {}
    neg_pair_counter = {}

    with open('./data/dataset_pair_text.txt') as fin:
        for line in fin:
            n += 1
            key = line[:-2].strip()
            if int(line[-2]) == 1:
                if key in pair_counter:
                    pair_counter[key] += 1
                else:
                    pair_counter[key] = 1
                pos += 1
            else:
                if key in neg_pair_counter:
                    neg_pair_counter[key] += 1
                else:
                    neg_pair_counter[key] = 1
                neg += 1

        keys = pair_counter.keys()
        neg_keys = neg_pair_counter.keys()
        intersec = list(set(keys).intersection(neg_keys))

    print('All pairs: {}'.format(n))

    print('Duplicates: {}'.format(pos - len(pair_counter)))
    # print(pair_counter)

    print('Negative Duplicates: {}'.format(neg - len(neg_pair_counter)))
    # print(neg_pair_counter)

    print('Contradictions: {}'.format(len(intersec)))
    # print(intersec)


def check_lem_duplicates():
    n = 0
    pair_counter = {}

    with open('./data/dataset_pair_text_lem.txt') as fin:
        for line in fin:
            n += 1
            key = line[:-2].strip()

            if key in pair_counter:
                pair_counter[key] += 1
            else:
                pair_counter[key] = 1

    print('Duplicates: {}'.format(n - len(pair_counter)))



if __name__ == '__main__':
    check_lem_duplicates()