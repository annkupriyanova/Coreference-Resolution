import xmltodict
import json


def make_item(xml_item):
    item = {
        'sh': int(xml_item['@sh']),
        'len': int(xml_item['@len']),
        'gram': xml_item['@gram'],
        'tok': xml_item['tok'],
        'lem': xml_item['lem']
    }
    if '@head' in xml_item:
        item['head'] = int(xml_item['@head'])

    return item


def make_group(xml_group, chain_id, var):
    group = {
        'group_id': int(xml_group['@group_id']),
        'chain_id': chain_id,
        'var': var,
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
    Parse groups.xml and write it to groups.json
    :param xml_file:
    :return: groups_dict
    """
    groups_dict = {}

    with open(xml_file, 'r') as fin:
        xml = xmltodict.parse(fin.read())

    for doc in xml['documents']['document']:
        doc_id = int(doc['@doc_id'])
        groups_dict[doc_id] = []

        for chn in doc['chains']['chain']:
            chain_id = int(chn['@chain_id'])
            var = int(chn['@var'])

            # chn['group'] is a dictionary
            if isinstance(chn['group'], dict):
                groups_dict[doc_id].append(make_group(chn['group'], chain_id, var))

            # chn['group'] is a list of dictionaries
            else:
                for grp in chn['group']:
                    groups_dict[doc_id].append(make_group(grp, chain_id, var))

    with open('groups.json', 'w') as fout:
        json.dump(groups_dict, fout, ensure_ascii=False, indent=2)

    return groups_dict


def main():
    print(parse_groups_xml('./RuCorData/groups.xml'))


if __name__ == "__main__":
    main()