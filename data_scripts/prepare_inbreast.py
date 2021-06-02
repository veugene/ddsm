import argparse
import csv
import os
import re

import h5py
import numpy as np
import plistlib
from scipy import ndimage
import SimpleITK as sitk
from skimage.draw import polygon
from skimage.morphology import (binary_closing,
                                binary_opening,
                                flood)
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(description=""
        "Create an HDF5 dataset with INbreast data. Sick images are those "
        "that contain masses and healthy images are those that do not. "
        "However, any images can contain calcifications. All images are "
        "cropped to remove irrelevant background and resized to a square.")
    parser.add_argument('path_inbreast', type=str,
                        help="path to the INbreast data directory")
    parser.add_argument('--path_create', type=str,
                        default='./data/in-breast/inbreast.h5',
                        help="path to save the HDF5 file to")
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    return parser




def load_inbreast_mask(mask_path, imshape=(4084, 3328)):
    """
    Shameless copy from:
    https://gist.github.com/Feyn-Man/de6f62997d051fc6ff75a6aa968537f5
    
    This function loads a osirix xml region as a binary numpy array for INBREAST
    dataset
    @mask_path : Path to the xml file
    @imshape : The shape of the image as an array e.g. [4084, 3328]
    return: numpy array where positions in the roi are assigned a value of 1.
    """

    mask = np.zeros(imshape)
    with open(mask_path, 'rb') as mask_file:
        plist_dict = plistlib.load(mask_file, fmt=plistlib.FMT_XML)['Images'][0]
        numRois = plist_dict['NumberOfROIs']
        rois = plist_dict['ROIs']
        assert len(rois) == numRois
        for roi in rois:
            numPoints = roi['NumberOfPoints']
            points = roi['Point_px']
            assert numPoints == len(points)
            points = [eval(point) for point in points]
            if len(points) <= 2:
                for point in points:
                    mask[int(point[1]), int(point[0])] = 1
            else:
                x, y = zip(*points)
                col, row = np.array(x), np.array(y) ##x coord is the column coord in an image and y is the row
                poly_x, poly_y = polygon(row, col, shape=imshape)
                mask[poly_x, poly_y] = 1
    return mask.astype(np.uint8)


def prepare_data_inbreast(args):
    target_dir = os.path.dirname(args.path_create)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Identify cases with masses from CSV.
    cases_with_mass = []
    with open(os.path.join(args.path_inbreast, "INbreast.csv"), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Mass ']=='X':
                cases_with_mass.append(row['File Name'])
    
    # Index cases by patient ID.
    image_files = {'h': {}, 's': {}}
    dicom_dir = os.path.join(args.path_inbreast, "AllDICOMs")
    for fn in os.listdir(dicom_dir):
        if not fn.endswith(".dcm"):
            continue # skip
        match = re.match(
            "^([0-9]{8})_([a-z0-9]{16})_MG_(L|R)_(CC|ML|FB)_ANON.dcm$", fn)
        assert match is not None
        case, patient, laterality, view = match.groups()
        if case in cases_with_mass:
            key = 's'
            assert os.path.exists(os.path.join(args.path_inbreast,
                                               "AllXML",
                                               f"{case}.xml"))
        else:
            key = 'h'
        image_files[key][case] = (os.path.join(dicom_dir, fn),
                                  patient,
                                  laterality,
                                  view)
    
    # Create HDF5 file to save dataset into.
    f = h5py.File(args.path_create, 'w')
    
    # Load healthy images and save in h5.
    print("Preparing healthy cases")
    for case in tqdm(sorted(image_files['h'].keys())):
        path, patient, laterality, view = image_files['h'][case]
        im_sitk = sitk.ReadImage(path)
        im = sitk.GetArrayFromImage(im_sitk)
        im = np.squeeze(im)
        im = trim(im)
        im = resize(im,
                     size=(args.resize, args.resize),
                     interpolator=sitk.sitkLinear)
        g_patient = f.require_group(patient)
        g_s = g_patient.require_group('s')
        g_s.create_dataset(case,
                           im.shape,
                           compression='lzf',
                           chunks=im.shape,
                           dtype=im.dtype)
        g_s.attrs.create('laterality', laterality)
        g_s.attrs.create('view', view)
    
    # Load sick images and masks and save in h5.
    print("Preparing sick cases")
    for case in tqdm(sorted(image_files['s'].keys())):
        path, patient, laterality, view = image_files['s'][case]
        im_sitk = sitk.ReadImage(path)
        im = sitk.GetArrayFromImage(im_sitk)
        shape = im.shape
        im = np.squeeze(im)
        m = load_inbreast_mask(os.path.join(args.path_inbreast,
                                            "AllXML", f"{case}.xml"),
                               imshape=im.shape)
        im, m = trim(im, m)
        im = resize(im,
                    size=(args.resize, args.resize),
                    interpolator=sitk.sitkLinear)
        m = resize(m,
                   size=(args.resize, args.resize),
                   interpolator=sitk.sitkNearestNeighbor)
        g_patient = f.require_group(patient)
        g_s = g_patient.require_group('s')
        g_s.create_dataset(case,
                           im.shape,
                           compression='lzf',
                           chunks=im.shape,
                           dtype=im.dtype)
        g_s.attrs.create('laterality', laterality)
        g_s.attrs.create('view', view)
        g_m = g_patient.require_group('m')
        g_m.create_dataset(case,
                           m.shape,
                           compression='lzf',
                           chunks=m.shape,
                           dtype=m.dtype)
        g_m.attrs.create('laterality', laterality)
        g_m.attrs.create('view', view)
        
        # DEBUG
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(im, cmap='gray')
        ax[1].imshow(m, cmap='gray')
        fig.savefig(os.path.join("debug_inbreast",
                                 os.path.basename(case)+".png"))
        plt.close()

    f.close()

def resize(image, size, interpolator=sitk.sitkLinear):
    sitk_image = sitk.GetImageFromArray(image)
    new_spacing = [x*y/z for x, y, z in zip(
                   sitk_image.GetSpacing(),
                   sitk_image.GetSize(),
                   size)]
    sitk_out = sitk.Resample(sitk_image,
                             size,
                             sitk.Transform(),
                             interpolator,
                             sitk_image.GetOrigin(),
                             new_spacing,
                             sitk_image.GetDirection(),
                             0,
                             sitk_image.GetPixelID())
    out = sitk.GetArrayFromImage(sitk_out)
    return out      

def trim(image, mask=None):
    if mask is not None:
        assert np.all(mask.shape==image.shape)
    
    # Normalize to within [0, 1] and threshold to binary.
    assert image.dtype==np.uint16
    x = image.copy()
    x_norm = x.astype(float)/(2**16 - 1)
    x[x_norm>=0.01] = 1
    x[x_norm< 0.01] = 0
    #x_bin = x.copy()
    
    # Align breast to left (find breast direction).
    # Check first 10% of image from left and first 10% from right. The breast
    # is aligned to the side with the highest cumulative intensity.
    l = max(int(x.shape[1]*0.1), 1)
    if x[:,-l:].sum() > x[:,:l].sum():
        image = np.fliplr(image)
        if mask is not None:
            mask = np.fliplr(mask)
        x = np.fliplr(x)
        x_norm = np.fliplr(x_norm)
    
    # Flood fill breast in order to then crop background out. Start flood 
    # at pixel 20% right from the left edge, center row.
    flood_row = max(x.shape[0]//2, 0)
    flood_col = max(int(x.shape[1]*0.2), 0)
    x_fill = binary_opening(x, selem=np.ones((5,5)))
    x_fill = binary_closing(x_fill, selem=np.ones((5,5)))
    m = flood(x_fill, (flood_row, flood_col), connectivity=1)
    
    # Crop out background outside of breast. Start 20% in from the left side
    # at the center row and move outward until the columns or rows are empty.
    crop_col_left, crop_col_right = 0, x.shape[1]
    crop_row_top, crop_row_bot = 0, x.shape[0]
    for col in range(flood_col, m.shape[1]):
        if not np.any(m[:,col]):
            crop_col_right = min(crop_col_right, col+1)
            break
    for row in range(flood_row, -1, -1):
        if not np.any(m[row,:]):
            crop_row_top = max(crop_row_top, row)
            break
    for row in range(flood_row, m.shape[0]):
        if not np.any(m[row,:]):
            crop_row_bot = min(crop_row_bot, row+1)
            break
    
    # Make sure crop row and column numbers are in range.
    crop_col_right = min(crop_col_right, image.shape[1])
    crop_row_top = max(crop_row_top, 0)
    crop_row_bot = min(crop_row_bot, image.shape[0])
    
    # Adjust crop to not crop mask (find mask bounding box).
    if mask is not None:
        slice_row, slice_col = ndimage.find_objects(mask>0, max_label=1)[0]    
        crop_col_left  = min(crop_col_left,  slice_col.start)
        crop_col_right = max(crop_col_right, slice_col.stop)
        crop_row_top   = min(crop_row_top,   slice_row.start)
        crop_row_bot   = max(crop_row_bot,   slice_row.stop)
    
    # Apply crop.
    image = image[crop_row_top:crop_row_bot,crop_col_left:crop_col_right]
    if mask is not None:
        mask = mask[crop_row_top:crop_row_bot,crop_col_left:crop_col_right]
        return image, mask
    return image


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    try:
        prepare_data_inbreast(args)
    except:
        raise