import json
import numpy as np

# DOC_GENRE = {}
EMBEDDING_DIM = 300

groups = {}
sorted_groups = {}
word_index = {}
embedding_matrix = []
group_index = {}

group_type = {}


def fillin_global_variables():
    global groups
    global sorted_groups
    global word_index
    global embedding_matrix
    global group_index

    with open('./data/groups_in_chains.json', 'r') as fin:
        groups = json.load(fin)

    with open('./data/sorted_groups.json') as fin:
        sorted_groups = json.load(fin)

    with open('/data/word_index_lemma.txt', 'r') as fin:
        for line in fin:
            line = line.split()
            word_index[line[0]] = int(line[1])

    embedding_matrix = np.load('/data/embedding_matrix_lemma.npy')

    with open('./data/group_index.txt', 'r') as fin:
        for line in fin:
            line = line.split()
            group_index[int(line[0])] = int(line[1])


def make_mention_features_dataset():

    fillin_global_variables()

    k = 0
    dataset = len(group_index)*[[0]]

    for doc_id, chains in groups.items():
        for chn in chains:
            for grp in chn['groups']:
                id = grp['group_id']
                embeddings = get_embeddings(grp)
                # add another features
                # ...

                dataset[group_index[id]] = embeddings
                k += 1
                # if dataset.size == 0:
                #     dataset = np.concatenate((id, embeddings))
                # else:
                #     dataset = np.vstack((dataset, np.concatenate((id, embeddings))))
        print("Doc {} is processed. {} groups is done".format(doc_id, k))

    dataset = np.array(dataset)
    print("Dataset is successfully collected. Congratulations!")

    np.save('/output/dataset_mention_features.npy', dataset)


# EMBEDDINGS

def get_embeddings(group):

    head = next((itm['lem'] for itm in group['items'] if 'head' in itm), group['items'][0]['lem'])
    head_embedding = get_head_embedding(head)

    words = [itm['lem'] for itm in group['items']]
    group_embedding = get_group_embedding(words)

    return np.concatenate((group_embedding, head_embedding))


def get_head_embedding(head):
    if head in word_index:
        return embedding_matrix[word_index[head]]
    else:
        return np.array(EMBEDDING_DIM*[0])


def get_group_embedding(words):
    embeddings = [ embedding_matrix[word_index[word]] if word.isalnum() else np.array(EMBEDDING_DIM*[0]) for word in words ]

    return np.mean(embeddings, axis=0)




# add morpho features later!!!!!!!
def get_mention_features(group, group_i, num_of_groups):
    """
    Get mention features vector.
    :param group: group in form of dictionary
    :param group_i: group index in the doc
    :param num_of_groups: number of groups in the doc
    :return: vector of features - str attr, ref attr, position in the doc and lenght in words
    """
    type_str = group_type['str'][group['attributes']['str']]
    type_ref = group_type['ref'][group['attributes']['ref']]
    position = group_i / num_of_groups
    length = len(group['items'])

    return np.array([type_str, type_ref, position, length])


def get_doc_genre(doc_id):

    return np.array([])


def get_distance_features(group1_i, group2_i):

    interven_groups = abs(group1_i - group2_i) - 1

    return np.array([interven_groups])


def string_match_features(group1, group2):
    """
    Get string match features vector
    :param group1:
    :param group2:
    :return: vector of features - exact match, head match and part match
    """
    lem_list1 = [item['lem'] for item in group1['items']]
    lem_list2 = [item['lem'] for item in group2['items']]

    head1 = next((item['lem'] for item in group1['items'] if 'head' in item), group1['items'][0]['lem'])
    head2 = next((item['lem'] for item in group2['items'] if 'head' in item), group2['items'][0]['lem'])

    exact_match = int(lem_list1 == lem_list2)
    head_match = int(head1 == head2)
    part_match = int(not not list(set(lem_list1) & set(lem_list2)))

    return np.array([exact_match, head_match, part_match])

### UTILITIES ###

def get_group_types():
    global group_type

    with open('group_types.txt', 'r') as fin:
        str_t = fin.readline().split()
        ref_t = fin.readline().split()

    group_type['str'] = {}
    group_type['ref'] = {}
    for index, t in enumerate(str_t):
        group_type['str'][t] = index
    for index, t in enumerate(ref_t):
        group_type['ref'][t] = index


def main():
    make_mention_features_dataset()
    return 0


if __name__ == "__main__":
    main()