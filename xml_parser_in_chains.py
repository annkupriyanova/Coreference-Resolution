import xmltodict
import json

group_str = []
group_ref = []
gram_values = []


def make_item(xml_item):
    global gram_values

    item = {
        'sh': int(xml_item['@sh']),
        'len': int(xml_item['@len']),
        'gram': xml_item['@gram'],
        'tok': xml_item['tok'],
        'lem': xml_item['lem']
    }
    if '@head' in xml_item:
        item['head'] = int(xml_item['@head'])

        gram_values.append(xml_item['@gram'])

    return item


def make_group(xml_group):
    global group_str
    global group_ref

    group = {
        'group_id': int(xml_group['@group_id']),
        'sh': int(xml_group['@sh']),
        'len': int(xml_group['@len']),
        'content': xml_group['content'],
        'items': []
    }
    if '@link' in xml_group:
        group['link'] = int(xml_group['@link'])

    if 'attributes' in xml_group:
        group['attributes'] = {}
        for attr in xml_group['attributes']['attr']:
            group['attributes'][attr['@name']] = attr['@val']

            if attr['@name'] == 'ref':
                group_ref.append(attr['@val'])
            elif attr['@name'] == 'str':
                group_str.append(attr['@val'])

    # grp_dict['items']['item'] is a dictionary
    if isinstance(xml_group['items']['item'], dict):
        group['items'].append(make_item(xml_group['items']['item']))

    # grp_dict['items']['item'] is a list of dictionaries
    else:
        for itm in xml_group['items']['item']:
            group['items'].append(make_item(itm))

    return group


def parse_groups_xml(xml_file):
    """
    Parse groups.xml and write it to groups_in_chains.json
    :param xml_file:
    :return: groups_dict
    """
    groups_dict = {}
    groups_position = {}

    with open(xml_file, 'r') as fin:
        xml = xmltodict.parse(fin.read())

    for doc in xml['documents']['document']:
        doc_id = int(doc['@doc_id'])
        groups_dict[doc_id] = []

        groups_position[doc_id] = []

        for chn in doc['chains']['chain']:
            chain = {
                'chain_id': int(chn['@chain_id']),
                'var': int(chn['@var']),
                'groups': []
            }

            # chn['group'] is a dictionary
            if isinstance(chn['group'], dict):
                group = make_group(chn['group'])
                chain['groups'].append(group)

                groups_position[doc_id].append((group['group_id'], chain['chain_id'], group['sh'], group['content']))

            # chn['group'] is a list of dictionaries
            else:
                for grp in chn['group']:
                    group = make_group(grp)
                    chain['groups'].append(group)

                    groups_position[doc_id].append(
                        (group['group_id'], chain['chain_id'], group['sh'], group['content']))

            groups_dict[doc_id].append(chain)

    with open('./data/groups_in_chains.json', 'w') as fout:
        json.dump(groups_dict, fout, ensure_ascii=False, indent=2)

    return groups_dict, groups_position


def make_group_types():
    """
    Generates file with group (mention) types - attribute 'str' and 'ref'
    :return: set of attribute 'str' values and attribute 'ref' values
    """
    global group_str
    global group_ref

    group_str = set(group_str)
    group_ref = set(group_ref)

    with open('./data/group_types.txt', 'w') as fout:
        fout.write(' '.join(group_str))
        fout.write('\n')
        fout.write(' '.join(group_ref))

    return group_str, group_ref


def make_gram_values():
    global gram_values

    gram_values = sorted(list(set(gram_values)))
    print(type(gram_values))

    with open('gram_values.txt', 'w') as fout:
        fout.write('\n'.join(gram_values))

    return gram_values


def make_sorted_groups(groups_position):
    for doc_id in groups_position:
        groups_position[doc_id] = sorted(groups_position[doc_id], key=lambda tup: tup[2])

    with open('./data/sorted_groups.json', 'w') as fout:
        json.dump(groups_position, fout, ensure_ascii=False)

    return groups_position


def make_group_index():
    group_index = {}

    with open('./data/sorted_groups.json') as fin:
        sorted_groups = json.load(fin)

    for _, groups in sorted_groups.items():
        k = len(group_index)
        for i, grp in enumerate(groups):
            group_index[grp[0]] = [i + k, -1]

    size = len(group_index)

    with open('./data/group_index.txt', 'w') as fout:
        for grp_id, i in group_index.items():
            # if i == size-1:
            #     fout.write("{} {} {}".format(grp_id, i[0], i[1]))
            # else:
            fout.write("{} {} {}\n".format(grp_id, i[0], i[1]))

    return group_index


def main():
    # _, groups_position = parse_groups_xml('./RuCorData/groups.xml')
    # print(make_group_types())
    # print(make_gram_values())
    # print(make_sorted_groups(groups_position))
    # make_word_index()
    make_group_index()


if __name__ == "__main__":
    main()
