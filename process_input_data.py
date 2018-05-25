import subprocess
import numpy as np
import itertools
from feature_extraction import get_group_embedding, get_head_embedding, \
    parse_gram_attribute, get_distance_features, fillin_global_variables
from nn_model import Model, PROB_THRESHOLD


def run_parser(input_file, output_file):
    bash_command = 'cd /Users/annakupriyanova/Desktop/parser/CoreNLP/target/classes'

    params = {
        'tagger': '/Users/annakupriyanova/Desktop/parser/russian-ud-pos.tagger',
        'taggerMF': '/Users/annakupriyanova/Desktop/parser/russian-ud-mf.tagger',
        'parser': '/Users/annakupriyanova/Desktop/parser/nndep.rus.modelMFWiki100HS400_80.txt',
        'mf': '',
        'pLemmaDict': '/Users/annakupriyanova/Desktop/parser/dict.tsv',
        'pText': input_file,
        'pResults': output_file
    }

    parser = 'java -Xmx8g edu.stanford.nlp.international.russian.process.Launcher '
    for k, val in params.items():
        parser += '-{} {} '.format(k, val)

    bash_command += '\n' + parser

    process = subprocess.Popen(bash_command, stdout=subprocess.PIPE, shell=True)
    output = process.communicate()[0].strip()

    print("Parsed file is ready")


def annotate(text):
    input_file = '/Users/annakupriyanova/PycharmProjects/CoreferenceResolution/input_data/input.txt'
    output_file = '/Users/annakupriyanova/PycharmProjects/CoreferenceResolution/input_data/output.conll'

    with open('./input_data/input.txt', 'w') as fout:
        fout.write(text)

    run_parser(input_file, output_file)

    return output_file


def get_conll(filepath):
    conll = [[]]
    i = 0

    with open(filepath, 'r') as fin:
        for line in fin:
            if line == '\n':
                conll.append([])
                i += 1
            else:
                row = line.split('\t')
                conll[i].append(row)

    return conll


def get_mentions(conll, indices):
    '''
    :param conll:
    :param indices:
    :return: list of mentions [[[group_lemmas], head_lemma, gram_str], ...]
    '''
    mentions = []
    word_forms = []

    for (sentNum, start, end, head) in indices:
        i = start
        group = []
        wf = []
        head_lemma = ''
        gram = ''

        while i <= end:
            token = conll[sentNum][i]

            lem = token[2]
            group.append(lem)
            wf.append(token[1])

            if i == head:
                head_lemma = lem
                gram = get_gram_string(token[4], token[5])

            i += 1

        mentions.append([group, head_lemma, gram])
        word_forms.append(' '.join(wf))

    return mentions, word_forms


def get_gram_string(pos, features):


    def morp_dict(features):
        d = {}

        for f in features:
            pair = f.split('=')
            d[pair[0]] = pair[1]

        return d

    gram = '------'
    morpo_features = features.split('|')

    if pos == 'NOUN':
        gram = 'N-'
    elif pos == 'ADJ':
        gram = 'A--'
    elif pos == 'PRON':
        gram = 'P--'

    morpho_dict = morp_dict(morpo_features)
    if 'Case' in morpho_dict:
        case = morpho_dict['Case'][0].lower()
    else:
        case = '-'

    if 'Gender' in morpho_dict:
        gender = morpho_dict['Gender'][0].lower()
    else:
        gender = '-'

    if 'Number' in morpho_dict:
        number = morpho_dict['Number'][0].lower()
    else:
        number = '-'

    gram += gender + number + case + '-'

    return gram


def prepare_data(mentions):
    fillin_global_variables()

    mentions_data = []
    data = []
    pairs = []

    for m in mentions:
        grp_embed = get_group_embedding(m[0])
        head_embed = get_head_embedding(m[1])
        morph = parse_gram_attribute(m[2])
        vec = np.r_[grp_embed, head_embed, morph]
        mentions_data.append(vec)

    indices = list(range(len(mentions)))

    for (i, j) in itertools.combinations(indices, 2):
        str_match = get_str_match_features(mentions[i][0], mentions[j][0], mentions[i][1], mentions[j][1])
        distance = get_distance_features(i, j)

        data.append(np.r_[mentions_data[i], mentions_data[j], str_match, distance])
        pairs.append((i, j))

    return np.array(data), pairs


def get_str_match_features(lem_list1, lem_list2, head1, head2):
    exact_match = int(lem_list1 == lem_list2)
    head_match = int(head1 == head2)
    part_match = int(not not list(set(lem_list1) & set(lem_list2)))

    return np.array([exact_match, head_match, part_match])


def predict(dataset):
    model = Model(mode='test', filepath='./models/CRModel_final.h5')
    print('Model is downloaded.')

    scores = model.predict(dataset)

    pred_labels = (scores > PROB_THRESHOLD)

    return scores, pred_labels


def print_results(pairs, labels, predictions, word_forms):
    for i, (m1, m2) in enumerate(pairs):
        print("{} {} {}".format((word_forms[m1], word_forms[m2]), labels[i], predictions[i]))


def CRPipeline(text, indices):
    out_file = annotate(text)

    conll = get_conll(out_file)

    mentions, word_forms = get_mentions(conll, indices)

    data, pairs = prepare_data(mentions)

    scores, pred_labels = predict(data)

    print_results(pairs, pred_labels, scores, word_forms)


def main():
    text = '''
    Привет.
    Меня зовут Аня.
    И я люблю пиццу.
    '''

    indices = [(1, 0, 0, 0), (1, 2, 2, 2), (2, 1, 1, 1), (2, 3, 3, 3)]

    # out_file = annotate(text)

    conll = get_conll('/Users/annakupriyanova/PycharmProjects/CoreferenceResolution/input_data/output.conll')

    mentions, word_forms = get_mentions(conll, indices)

    data, pairs = prepare_data(mentions)

    scores, pred_labels = predict(data)

    print_results(pairs, pred_labels, scores, word_forms)


if __name__ == '__main__':
    main()