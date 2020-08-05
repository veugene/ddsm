import argparse
import csv
import ctypes
from functools import partial
import os
import multiprocessing
from subprocess import call
import tempfile

from numpy import array, savetxt
import numpy as np
import SimpleITK as sitk


####################################################
# Process arguments
####################################################
def parse_args():
    parser = argparse.ArgumentParser(description=""
                "Make a tiff dataset from DDSM raws of normal cases.")
    parser.add_argument('read_from', type=str,
                        help="the path to the DDSM normals directory")
    parser.add_argument('write_to', type=str,
                        help="the destination directory for the converted "
                             "(tiff) files")
    parser.add_argument('--resize', type=int, default=None,
                        help="resize images to a square of this size")
    parser.add_argument('--force', action='store_true',
                        help="force overwrite of existing files")
    args = parser.parse_args()
    return args


####################################################
# CSV fields to record
####################################################
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
                print(row_name, idx)
                print(lst)
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
        lines = list(map(lambda s: s.strip().split(), f.readlines()))

    # map ics data to values
    ics_dict = {
        'patient_id': get_value(lines, 'filename', 1),
        'age': get_value(lines, 'PATIENT_AGE', 1),
        'scanner_type': get_value(lines, 'DIGITIZER', 1),
        'scan_institution': scanner_map[(letter, get_value(lines,
                                                           'DIGITIZER', 1))],
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


####################################################
# Process an image from a normal DDSM case
####################################################
class ddsm_normal_case_image(object):
    def __init__(self, path, ics_dict):

        fname = os.path.basename(path)
        case_id, sequence, ext = fname.split('.')
        self.case_id = case_id
        self.sequence = sequence
        self.ext = ext

        # file path data
        self.path = path

        # image information
        self.height = ics_dict[sequence]['height']
        self.width = ics_dict[sequence]['width']
        self.bpp = ics_dict[sequence]['bpp']
        self.resolution = ics_dict[sequence]['resolution']
        self.scanner_type = ics_dict['scanner_type']
        self.scan_institution = ics_dict['scan_institution']

        # patient information
        self.patient_id = ics_dict['patient_id']
        self.side, self.view = sequence.split('_')
        self.breast_density = ics_dict['density']

        # image information
        self._raw_image = None

    def _decompress_ljpeg(self, log_file_path='ljpeg_decompression_log.txt'):
        """
        :param im_path: base path for ljpeg
        :param log_file_path: path to log for writing these
        :return: None
        """
        with open(log_file_path, 'a') as log_file:
            # Assume jpeg binary is in PATH
            # TODO: install jpeg to a binary directory with setup.py
            call_lst = ['jpeg', '-d', '-s', self.path]
            call(call_lst, stdout=log_file)

    def _read_raw_image(self, force=False):
        """
        Read in a raw image into a numpy array
        :param force: boolean flag if we should force a read if we already have this image
        :return: None
        """

        # only read if we haven't already or
        # we aren't trying to force a read
        if (self._raw_image is not None) and not force:
            return

        try:
            # make sure decompressed image exists
            self._decompress_ljpeg()

            # read it in and make it correct
            raw_im_path = "{}.1".format(self.path)
            im = np.fromfile(raw_im_path, dtype=np.uint16)
            im.shape = (self.height, self.width)
            self._raw_image = im.byteswap()  # switch endian
        finally:
            # clean up : delete decompressed ljpeg
            os.remove("{}.1".format(self.path))

    def _od_correct(self, im):
        """
        Map gray levels to optical density level
        :param im: image
        :return: optical density image
        """
        im_od = np.zeros_like(im, dtype=np.float64)
        MAXVAL = 2**16-1
        if (self.scan_institution == 'MGH') and (self.scanner_type == 'DBA'):
            im_clip = np.clip(im,0,MAXVAL-1)+1 # add 1 to keep from log(0)
            im_od = (np.log10(im_clip) - 4.80662) / -1.07553
        elif (self.scan_institution == 'MGH') and (self.scanner_type == 'HOWTEK'):
            im_od = (-0.00094568 * im) + 3.789
        elif (self.scan_institution == 'WFU') and (self.scanner_type == 'LUMISYS'):
            im_od = (im - 4096.99) / -1009.01
        elif (self.scan_institution == 'ISMD') and (self.scanner_type == 'HOWTEK'):
            im_od = (-0.00099055807612 * im) + 3.96604095240593

        # perform heath noise correction
        im_od[im_od < 0.05] = 0.05
        im_od[im_od > 3.0] = 3.0
        return im_od

    def save_image(self,
                   out_dir,
                   out_name=None,
                   od_correct=False,
                   make_dtype=None,
                   resize=None,
                   force=False):
        """
        save the image data as a tiff file (without correction)
        :param out_dir: directory to put this image
        :param out_name: name of file to save image as
        :param od_correct: boolean to decide to perform od_correction
        :param make_dtype: boolean to switch to 8-bit encoding
        :param force: force if this image already exists
        :return: path of the image
        """
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        # construct image path
        out_name = os.path.split(self.path)[1].replace(".LJPEG", ".tif")
        im_path = os.path.join(out_dir, out_name)

        # don't write if image exists and we aren't forcing it
        if os.path.exists(im_path) and not force:
            return im_path

        # make sure we have an image to save
        if self._raw_image is None:
            self._read_raw_image()

        im_array = np.copy(self._raw_image)

        # convert to optical density
        if od_correct:
            im_array = self._od_correct(im_array)
            im_array = np.interp(im_array, (0.0, 4.0), (255, 0))
            im_array = im_array.astype(np.uint8)

        if make_dtype == 'uint8':
            pass

        # create image object
        im = sitk.GetImageFromArray(im_array)

        # resize if necessary
        if resize:
            im = sitk.Resample(im,
                               (resize, resize),
                               sitk.Transform(),
                               sitk.sitkLinear,
                               im.GetOrigin(),
                               im.GetSpacing(),
                               im.GetDirection(),
                               0,
                               im.GetPixelID())

        # save image (make this atomic in case of interruption)
        tmp = tempfile.NamedTemporaryFile(delete=False,
                                          dir=out_dir,
                                          suffix='.tif')
        tmp_path = os.path.join(out_dir, tmp.name)
        sitk.WriteImage(im, tmp_path, True)  # True for useCompression
        os.rename(tmp_path, im_path)         # Atomic move

        # return location of image
        return im_path


####################################################
# Create the dataset (read, convert, save)
####################################################
def make_data_set(read_from, write_to, resize=None, force=False):
    if not os.path.exists(write_to):
        os.makedirs(write_to)
    
    # Walk through image directory tree; load ics files and note down
    # all image paths for later multiprocessing.
    image_list = []     # tuples of (image_path, ics_dict)
    for curdir, dirs, files in os.walk(read_from):
        raw_image_filenames = []
        ics_filename = None
        for f in files:
            if f.endswith('.LJPEG'):
                raw_image_filenames.append(f)
            elif f.endswith('.ics'):
                ics_filename = f
        if not ics_filename:
            continue
        
        # Read ics info.
        ics_path = os.path.join(curdir, ics_filename)
        ics_dict = get_ics_info(ics_path)
        
        # Record paths.
        image_list.extend([(os.path.join(curdir, fn), ics_dict)
                           for fn in raw_image_filenames])
    
    # Use multiprocessing to convert images.
    args = (read_from, write_to, resize, force)
    with multiprocessing.Pool() as pool:
        log = pool.map(partial(convert_image,
                               read_from=read_from,
                               write_to=write_to,
                               resize=resize,
                               force=force),
                       image_list)
    
    # Parse results. Record output in csv file.
    outfile = open(os.path.join(write_to, 'ddsm_normal_cases.csv'), 'w')
    outfile_writer = csv.writer(outfile, delimiter=',')
    outfile_writer.writerow(fields)
    count_success = 0
    count_failure = 0
    for result in log:
        if result[0]==0:
            count_success += 1
            outfile_writer.writerow(result[1])
        else:
            count_failure += 1
    outfile.close()
    
    # Report on the number of successes and failures.
    print("SUCCESS: {}, FAILURE: {}".format(count_success, count_failure))


####################################################
# Convert a raw image to a tif (and save)
####################################################
def convert_image(image_tuple, read_from, write_to, resize, force):
    path, ics_dict = image_tuple
    rel_path = os.path.relpath(path, read_from)
    case = ddsm_normal_case_image(path, ics_dict)
    dir_write_to = os.path.join(write_to, os.path.dirname(rel_path))
    print("Converting {} ... ".format(rel_path), end="")
    try:
        # uint8 optical density
        save_path = case.save_image(out_dir=dir_write_to,
                                    od_correct=True,
                                    resize=resize,
                                    force=force)
        case.od_img_path = save_path    # to save in csv
        print("done")
        return (0, [getattr(case, f) for f in fields])
    except Exception as e:
        print("error : {}".format(e))
    return (1, None)


if __name__=='__main__':
    args = parse_args()
    make_data_set(read_from=args.read_from,
                  write_to=args.write_to,
                  resize=args.resize,
                  force=args.force)