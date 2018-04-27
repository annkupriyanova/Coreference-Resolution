import json


sorted_groups = {}
group_index = {}


def fillin_global_variables():
    global groups, sorted_groups
    global group_index


    with open('./data/sorted_groups.json') as fin:
        sorted_groups = json.load(fin)

    with open('./data/group_index.txt') as fin:
        for line in fin:
            line = line.split()
            group_index[int(line[0])] = [int(line[1]), int(line[2])]


def get_sentence_context():
    global group_index
    sent_index = {}

    docs = get_docs()

    for doc_id, groups in sorted_groups.items():
        text = docs[doc_id]
        sent_i = 0
        # sent_index_len = len(sent_index)
        if sent_index:
            max_key = max(sent_index, key=int)
        else:
            max_key = 0

        for grp in groups:
            found = None
            while sent_i < len(text) and not found:
                # condition: tok[shift] == grp[sh]
                found = next((tok for tok in text[sent_i] if tok[0] == grp[2]), None)
                sent_i += 1

            if found:
                sent_i -= 1
                # add sentence
                idx = sent_i + 1 + max_key
                if idx not in sent_index:
                    # sent_index[idx] = ' '.join([tok[2] for tok in text[sent_i] if tok[2].isalnum()])
                    sent_index[idx] = get_sentence(text[sent_i])
                group_index[grp[0]][1] = idx

    with open('./data/sent_index.txt', 'w') as fout:
        json.dump(sent_index, fout, ensure_ascii=False, indent=2)

    # update group_index with sentence indices
    with open('./data/group_index.txt', 'w') as fout:
        for grp_id, i in group_index.items():
            fout.write("{} {} {}\n".format(grp_id, i[0], i[1]))

    return sent_index


def get_docs():
    docs = {}
    sent_i = 0

    with open('./data/Tokens.txt') as fin:
        for line in fin:
            vals = line.split('\t')

            if vals[0] != 'doc_id':
                # new doc_id
                if vals[0] not in docs:
                    docs[vals[0]] = [[]]
                    sent_i = 0

                # append [shift, token, lemma]
                docs[vals[0]][sent_i].append([int(vals[1]), vals[3], vals[4], vals[5]])

                # not end of sentence
                if vals[5] == 'SENT\n':
                    sent_i += 1
                    docs[vals[0]].append([])

    return docs


def get_sentence(sent):
    sentence = []

    for tok in sent:
        if tok[3] == 'SENT\n':
            tok[2] = ''.join(c for c in tok[2] if c.isalpha())
            if tok[2] != '':
                sentence.append(tok[2])

        elif tok[2].isalpha():
            sentence.append(tok[2])

    sentence = ' '.join(sentence)

    return sentence


def main():
    fillin_global_variables()
    get_sentence_context()


if __name__ == '__main__':
    main()
