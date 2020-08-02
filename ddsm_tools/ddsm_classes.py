from numpy import array, savetxt
import numpy as np
import os
from subprocess import call
from PIL import Image, ImageDraw

# hack because PIL doesn't like uint16
Image._fromarray_typemap[((1, 1), "<u2")] = ("I", "I;16")


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

    ###################################################
    # Image Methods
    ###################################################
    def _decompress_ljpeg(self, log_file_path='ljpeg_decompression_log.txt'):
        """
        :param im_path: base path for ljpeg
        :param log_file_path: path to log for writing these
        :return: None
        """
        with open(log_file_path, 'a') as log_file:
            if os.path.exists("{}.1".format(self.path)):
                log_file.write("Decompressed LJPEG Exists: {}"
                               "".format(self.path))
            else:
                call_lst = ['./jpegdir/jpeg', '-d', '-s', self.path]
                call(call_lst, stdout=log_file)
        print("Decompressed {}".format(self.path))

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

        # make sure decompressed image exists
        raw_im_path = "{}.1".format(self.path)
        if not os.path.exists(raw_im_path):
            self._decompress_ljpeg()

        # read it in and make it correct
        im = np.fromfile(raw_im_path, dtype=np.uint16)
        im.shape = (self.height, self.width)
        self._raw_image = im.byteswap()  # switch endian

    def _od_correct(self, im):
        """
        Map gray levels to optical density level
        :param im: image
        :return: optical density image
        """
        im_od = np.zeros_like(im, dtype=np.float64)

        if (self.scan_institution == 'MGH') and (self.scanner_type == 'DBA'):
            im_od = (np.log10(im + 1) - 4.80662) / -1.07553  # add 1 to keep from log(0)
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

    # todo fix output paths
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
        # construct image path
        out_name = os.path.split(self.path).replace(".LJPEG", ".tif")
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
        im = Image.fromarray(im_array)

        # resize if necessary
        if resize:
            im = im.resize(resize, resample=Image.LINEAR)

        # save image
        im.save(im_path, 'tiff')

        # return location of image
        return im_path
