import json
from os.path import exists
import string

groups = {}
sorted_groups = {}
group_index = {}
documents = {}


def fillin_global_variables():
    global groups, sorted_groups
    global group_index
    global documents

    with open('./data/groups_in_chains.json') as fin:
        groups = json.load(fin)

    with open('./data/sorted_groups.json') as fin:
        sorted_groups = json.load(fin)

    with open('./data/group_index.txt') as fin:
        for line in fin:
            line = line.split()
            group_index[int(line[0])] = int(line[1])

    with open('./RuCorData/Documents.txt') as fin:
        for line in fin:
            values = line.split()
            documents[values[0]] = values[1][:-4] + '.conll'


# def get_sentence(text, index):
#     STOPS = ['.', '?', '!']
#
#     # go to the left from index
#     left = index
#     while left > 0 and text[left] not in STOPS:
#         left -= 1
#     # if text[left + 1] == ' ':
#     #     left += 2
#
#     # go to the right from index
#     right = index
#     while right < len(text) and text[right] not in STOPS:
#         right += 1
#
#     sent = text[left:right]
#
#     return sent, left


def get_sentence_context():
    context = len(group_index) * [[0]]

    for doc_id, groups in sorted_groups.items():
        filepath = './RuCorData/parsed_testset/' + documents[doc_id]

        if exists(filepath):
            text = get_text(filepath)

            sent_i = 0

            for grp in groups:
                content = grp[3]
                result = None
                while sent_i < len(text) and not result:
                    result = check_group_in_sent(content, text[sent_i])
                    sent_i += 1

                if result:
                    sent_i -= 1
                    idx = group_index[grp[0]]
                    context[idx] = ' '.join([tok[1] for tok in text[sent_i] if tok[1].isalnum()])

        else:
            print('Doc {} is not parsed =('.format(doc_id))
            # !!!!!!!make something with unparsed texts
            continue

        print('Doc {} is processed.'.format(doc_id))

    with open('./data/context_sentence.txt', 'w') as fout:
        for con in context:
            fout.write('{}\n'.format(con))

    return context

    # if exists(filepath):
    #     with open(filepath) as fin:
    #         text = fin.read()
    # else:
    #     continue
    #     table = str.maketrans({key: None for key in string.punctuation})
    #     text = text.translate(table)
    #     text = text.split()
    #
    #     for grp in groups:
    #         content = grp[3]
    #         # find index of the group beginning
    #         i = text.index(content) if content in text else -1
    #         if i > -1:
    #             # get sentence with this group
    #             sent, left = get_sentence(text, i)
    #             # add this sentence to context matrix
    #             context[group_index[grp[0]]] = sent
    #             # cut text from the beginning till the beginning of group sentence
    #             text = text[left:]
    #


def get_text(filepath):
    text = []
    sent = []
    with open(filepath) as fin:
        for line in fin:
            if line != '\n' and line != '':
                values = line.split('\t')
                # append tuple (tok,lem) to sentence
                sent.append((values[1], values[2]))
            elif line == '\n':
                text.append(sent)
                sent = []
    return text


def check_group_in_sent(content, sentence):
    if len(content.split()) == 1:
        result = next((tok for tok in sentence if tok[0] == content), None)
    else:
        content_first = content.split()[0]
        result = next((tok for tok in sentence if tok[0] == content_first), None)

    return result


def main():
    fillin_global_variables()
    # get_sentence_context()
    find_unparsed_docs()


def find_unparsed_docs():
    docs = []
    for doc_id, groups in sorted_groups.items():
        filepath = './RuCorData/parsed_testset/' + documents[doc_id]

        if not exists(filepath):
            docs.append(documents[doc_id])
    print(docs)
    return docs


if __name__ == '__main__':
    main()
