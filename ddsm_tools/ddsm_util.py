import os

####################################################
# Extract value from split list of data
####################################################
def get_value(lst, row_name, idx):
    """
    :param lst: data list, each entry is another list with whitespace separated data
    :param row_name: name of the row to find data
    :param idx: numeric index of desired value
    :return: value
    """
    val = None
    for l in lst:
        if not l:
            continue

        if l[0] == row_name:
            try:
                val = l[idx]
            except Exception:
                print row_name, idx
                print lst
                val = None
            break
    return val


####################################################
# ICS Information extraction
####################################################
scanner_map = {
    ('A', 'DBA'): 'MGH',
    ('A', 'HOWTEK'): 'MGH',
    ('B', 'LUMISYS'): 'WFU',
    ('C', 'LUMISYS'): 'WFU',
    ('D', 'HOWTEK'): 'ISMD'
}


def get_ics_info(ics_file_path):
    """
    :param ics_file_path: path to ics file
    :return: dictionary containing all relevant data in ics file
    """
    # get letter for scanner type
    ics_file_name = os.path.basename(ics_file_path)
    letter = ics_file_name[0]

    # get data from ics file
    with open(ics_file_path, 'r') as f:
        lines = map(lambda s: s.strip().split(), f.readlines())

    # map ics data to values
    ics_dict = {
        'patient_id': get_value(lines, 'filename', 1),
        'age': get_value(lines, 'PATIENT_AGE', 1),
        'scanner_type': get_value(lines, 'DIGITIZER', 1),
        'scan_institution': scanner_map[(letter, get_value(lines, 'DIGITIZER', 1))],
        'density': get_value(lines, 'DENSITY', 1)
    }

    for sequence in ['LEFT_CC', 'RIGHT_CC', 'LEFT_MLO', 'RIGHT_MLO']:
        if get_value(lines, sequence, 0) is None:
            continue

        sequence_dict = {
            'height': int(get_value(lines, sequence, 2)),
            'width': int(get_value(lines, sequence, 4)),
            'bpp': int(get_value(lines, sequence, 6)),
            'resolution': float(get_value(lines, sequence, 8))
        }

        ics_dict[sequence] = sequence_dict

    return ics_dict
