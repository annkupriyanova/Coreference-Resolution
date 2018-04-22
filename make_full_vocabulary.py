from gensim.models.fasttext import FastText
import numpy as np

EMBEDDING_DIM = 300


def get_list_tokens():
    tokens_lem = []
    tokens = []

    with open('./data/Tokens.txt', 'r') as fin:
        _ = fin.readline()

        for line in fin:
            tok = line.split('\t')[3]
            lem = line.split('\t')[4]
            if tok.isalnum():
                tokens.append(tok)
                tokens_lem.append(lem)

    tokens = list(set(tokens))
    tokens_lem = list(set(tokens_lem))

    print('Tokens: ', len(tokens))
    print('Lemmas: ', len(tokens_lem))

    return tokens, tokens_lem


def make_word_index(tokens, filename):
    word_index = {}

    for i, tok in enumerate(tokens):
        word_index[tok] = i

    size = len(word_index)

    with open(filename, 'w') as fout:
        for tok, i in word_index.items():
            if i == size - 1:
                fout.write(tok + ' ' + str(i))
            else:
                fout.write(tok + ' ' + str(i) + '\n')

    return word_index


def make_pretrained_embeddings(word_index):
    model = FastText.load_fasttext_format('/models/wiki.ru')

    # create embedding_matrix: "index of word - vector"
    embedding_matrix = np.zeros((len(word_index), EMBEDDING_DIM))
    for lem, i in word_index.items():
        if lem in model:
            embedding_matrix[i] = model[lem]

    # write embedding_matrix to file in the output directory of Floyd
    np.save('/output/embedding_matrix_lemma.npy', embedding_matrix)

    return embedding_matrix


def get_word_index(filename):
    word_index = {}

    with open(filename, 'r') as fin:
        for line in fin:
            line = line.split()
            word_index[line[0]] = int(line[1])

    return word_index


# def check_zero_vecs():
#
#     word_index = get_word_index('./data/word_index_lemma.txt')
#     embedding_matrix = np.load('./data/embedding_matrix_lemma.npy')
#
#     print("Check if embedding_matrix has zero vectors. Zero tokens: ")
#     ind = []
#     for i, vec in enumerate(embedding_matrix):
#         all_zeros = not np.any(vec)
#         if all_zeros:
#             ind.append(i)
#
#     if not ind:
#         print("No empty vectors!")
#
#     for i in ind:
#         tok = next((t for t in word_index if word_index[t] == i), None)
#         print(tok)


def main():
    _, tokens_lem = get_list_tokens()
    print("Tokens have been collected.")

    word_index = make_word_index(tokens_lem, '/output/word_index_lemma.txt')
    print("Word index has been made.")

    embedding_matrix = make_pretrained_embeddings(word_index)
    print("Embedding matrix has been made.")


if __name__ == '__main__':
    main()
