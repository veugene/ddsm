import csv
import os

from ddsm_util import get_ics_info, get_num_abnormalities
from ddsm_classes import ddsm_normal_case_image

fields = ['patient_id',
          'breast_density',
          'side',
          'view',
          'scanner_type',
          'scan_institution',
          'width',
          'height',
          'bpp',
          'resolution',
          'od_img_path']


def make_data_set(read_from, write_to, resize=None):
    outfile = open(os.path.join(write_to, 'ddsm_normal_cases.csv'), 'w')
    outfile_writer = csv.writer(outfile, delimiter=',')
    outfile_writer.writerow(fields)

    # Walk through image files; load and convert normal cases.
    count = 0
    for curdir, dirs, files in os.walk(read_from):
        # Per case, collect all raw image files and the ics file.
        raw_image_files = []
        ics_file_path = None
        for f in files:
            if f.endswith('.LJPEG'):
                raw_image_files.append(os.path.join(root, curdir, f))
            elif f.endswith('.ics'):
                ics_file_path = os.path.join(root, curdir, f)
        if not ics_file_path:
            continue
        
        # Read ics info.
        ics_dict = get_ics_info(ics_file_path)
        
        # Convert each raw file.
        for filename in raw_image_files:
            case = ddsm_normal_case_image(
                os.path.join(read_from, curdir, filename),
                ics_dict)
            dir_write_to = os.path.join(write_to, curdir)
            try:
                # uint8 optical density
                save_path = case.save_image(out_dir=dir_write_to,
                                            od_correct=True,
                                            resize=resize)
            except ValueError:
                print("Error with case {}".format(case.path))
            try:
                outfile_writer.writerow(
                    [getattr(abnormality, f) for f in fields])
            except AttributeError:
                print("Abnormality {} has no od image"
                        "".format(abnormality.path))
            print("Converted {}".format(save_path))

    outfile.close()
