from nn_model import Model
from feature_extraction import get_group_embedding, get_head_embedding, parse_gram_attribute, fillin_global_variables
import numpy as np
import itertools


PROB_THRESHOLD = 0.7


def prepare_samples():
    """
    1. Кольца Урана — система планетных колец, окружающих Уран.
    Среди других систем колец она занимает промежуточное положение по сложности строения между более развитой системой
    колец Сатурна и простыми системами колец Юпитера и Нептуна.
    2. Софья Андреева (эта восемнадцатилетняя дворовая, то есть мать моя) была круглою сиротою уже несколько лет;
    покойный же отец ее...
    """

    dataset_sample = 6 * [[0]]
    dataset = []
    pairs = []

    fillin_global_variables()

    grps = ['кольца уран', 'система планетный кольцо', 'она', 'промежуточный положение', 'более развитой система кольцо сатурн']
    heads = ['кольца', 'система', 'она', 'положение', 'система']
    grams = ['Ncmsny', 'Ncfsnn', 'P-3fsnn', 'Ncnsnn', 'Ncfsin']

    # grps = ['софья андреев', 'этот восемнадцатилетняя дворовый', 'мать мой', 'круглою сиротою', 'покойный отец', 'она']
    # heads = ['софья', 'дворовый', 'мать', 'сиротою', 'отец', 'она']
    # grams = ['Npfsny', 'Afpfsnf', 'Ncfsny', 'Ncfsin', 'Ncmsna', 'P-3fsan']

    for i, grp in enumerate(grps):
        head_embedding = get_head_embedding(heads[i])
        group_embedding = get_group_embedding(grp.split())
        embeddings = np.concatenate((group_embedding, head_embedding))

        morph_features = parse_gram_attribute(grams[i])

        dataset_sample[i] = np.r_[embeddings, morph_features]

    dataset_sample = np.asarray(dataset_sample)
    print("Dataset of samples id ready.")

    indices = [i for i, _ in enumerate(grps)]

    for (i, j) in itertools.combinations(indices, 2):
        dataset.append(np.r_[dataset_sample[i], dataset_sample[j]])
        pairs.append((grps[i], grps[j]))

    dataset = np.asarray(dataset)
    print("Dataset is ready.")

    return dataset_sample, dataset, pairs


def print_results(pairs, labels, predictions):
    for i, p in enumerate(pairs):
        print("{} {} {}".format(p, labels[i], predictions[i]))

def main():

    _, dataset, pairs = prepare_samples()

    model = Model(mode='test', filepath='./models/CRModel_final.h5')
    print('Model is downloaded.')

    predictions = model.predict(dataset)
    # print("Predictions: {}".format(predictions))

    pred_labels = (predictions > PROB_THRESHOLD)
    # print("Predicted labels: {}".format(pred_labels))

    print_results(pairs, pred_labels, predictions)

    # true_labels = [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0]
    true_labels = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0]


    model.estimate_metrics(dataset, true_labels)

if __name__ == '__main__':
    main()