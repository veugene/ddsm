import argparse
import os
import re

#import cv2
import h5py
import numpy as np
from scipy import ndimage
import SimpleITK as sitk
from skimage import transform
from skimage.morphology import (binary_closing,
                                binary_opening,
                                flood)
from tqdm import tqdm

from data_tools.io import h5py_array_writer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path_cbis', type=str,
                        help="path to the CBIS-DDSM data directory")
    parser.add_argument('--path_create', type=str,
                        default='./data/ddsm/cbis_ddsm.h5',
                        help="path to save the HDF5 file to")
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--min_mask_pixels', type=int, default=25)
    return parser.parse_args()


def create_dataset(args):
    
    ##
    # Record all mask files per case and their sizes.
    ##
    mask_paths = []
    for root, dirs, files in os.walk(args.path_cbis):
        if re.search("^.*Mass-(Training|Test).*(CC|MLO)_[1-9]\/.*$", root):
            for fn in files:
                # Only look at dicom files.
                if fn.endswith(".dcm"):
                    mask_paths.append(os.path.join(root, fn))
    
    print("Indexing masks")
    mask_path_index = {}
    for path in tqdm(mask_paths):   
        # The case this mask is associated with.
        case = re.search("^.*Mass-(Training|Test).*(CC|MLO)", path).group(0)
        
        # Check the mask's size.
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        reader.ReadImageInformation()
        x, y, z = reader.GetSize()
        
        # Record.
        if case not in mask_path_index:
            mask_path_index[case] = []
        mask_path_index[case].append((path, x, y))
    
    ##
    # Filter out all masks whose sizes do not match the case image.
    ##
    image_paths = []
    for case in sorted(mask_path_index.keys()):
        paths = []
        for root, dirs, files in os.walk(case):
            dcm_files = [os.path.join(root, fn) for fn in files
                         if fn.endswith(".dcm")]
            paths.extend(dcm_files)
        assert len(paths)==1
        image_paths.append((case, paths[0]))
    
    print("Indexing images")
    masks_by_case = {}
    for case, image_path in tqdm(image_paths):
        # Check the image's size.
        image_path = image_path
        reader = sitk.ImageFileReader()
        reader.SetFileName(image_path)
        reader.ReadImageInformation()
        x_im, y_im, z_im = reader.GetSize()
        
        # Compare to mask sizes.
        for mask_path, x, y in mask_path_index[case]:
            if x==x_im and y==y_im:
                if case not in masks_by_case:
                    masks_by_case[case] = []
                masks_by_case[case].append(mask_path)
    
    # Create h5py writers.
    writer_kwargs = {'data_element_shape': (256, 256),
                     'batch_size': args.batch_size,
                     'filename': args.path_create,
                     'append': True,
                     'kwargs': {'chunks': (args.batch_size, 256, 256)}}
    writer = {'train': {}, 'test': {}}
    for fold in ['train', 'test']:
        writer[fold]['s'] = h5py_array_writer(
            array_name='{}/{}'.format(fold, 's'),
            dtype=np.uint16,
            **writer_kwargs)
        writer[fold]['m'] = h5py_array_writer(
            array_name='{}/{}'.format(fold, 'm'),
            dtype=np.uint8,
            **writer_kwargs)
    
    # Save images and masks. Combine masks for each case.
    print("Processing and saving images and masks")
    for case, image_path in tqdm(image_paths):
        if case not in masks_by_case:
            # No masks for this case.
            print("no masks for {}; skipping".format(case))
            continue
        fold = 'train'
        if re.search("Test", case):
            fold = 'test'
        
        # Read image.
        image_sitk = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(image_sitk)
        
        # Read and combine masks.
        combined_mask = None
        for mask_path in masks_by_case[case]:
            mask = sitk.ReadImage(mask_path)
            mask_np = sitk.GetArrayFromImage(mask)
            if combined_mask is None:
                combined_mask = np.zeros(mask_np.shape, dtype=np.uint8)
                combined_mask[mask_np>0] = 1
            else:
                print("DEBUG", case,
                      re.match("^.*Mass-(?:Training|Test).*(?:CC|MLO)_([1-9])\/.*$", mask_path).group(1),
                      np.count_nonzero(np.logical_and(combined_mask,
                                                      mask_np)),
                      np.count_nonzero(np.logical_or(combined_mask,
                                                     mask_np)),
                      np.count_nonzero(combined_mask),
                      np.count_nonzero(mask_np))
                combined_mask[np.logical_or(combined_mask, mask_np)] = 1
        
        # Remove redundant dimension.
        image = np.squeeze(image)
        combined_mask = np.squeeze(combined_mask)
        
        # Trim image and mask to breast.
        image, combined_mask = trim(image, combined_mask)
        
        # Resize image and mask.
        image = resize(
            image,
            size=(args.resize, args.resize),
            interpolator=sitk.sitkLinear)
        combined_mask = resize(
            combined_mask,
            size=(args.resize, args.resize),
            interpolator=sitk.sitkNearestNeighbor)
        
        # Check if there are enough pixels in the mask. If not, skip.
        n_pixels = np.count_nonzero(combined_mask)
        if n_pixels < args.min_mask_pixels:
            print("{} pixels in mask for {}; skipping"
                  "".format(n_pixels, case))
        
        # Write
        writer[fold]['s'].buffered_write(image)
        writer[fold]['m'].buffered_write(combined_mask)
    
    # Flush writers.
    for fold in ['train', 'test']:
        writer[fold]['s'].flush_buffer()
        writer[fold]['m'].flush_buffer()


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


#def resize(image, size, interpolation=cv2.INTER_AREA):
    #out = cv2.resize(image,
                     #dsize=size,
                     #interpolation=interpolation)
    #return out


#def resize(image, size, interpolator=None):
    #dtype = image.dtype
    #out = transform.resize(image,
                           #output_shape=size,
                           #mode='constant',
                           #cval=0,
                           #clip=True,
                           #preserve_range=True,
                           #anti_aliasing=True)
    #return out.astype(dtype)


def trim(image, mask=None):
    if mask is not None:
        assert np.all(mask.shape==image.shape)
    
    # Normalize to within [0, 1] and threshold to binary.
    assert image.dtype==np.uint16
    x = image.copy()
    x_norm = x.astype(np.float)/(2**16 - 1)
    x[x_norm>=0.075] = 1
    x[x_norm< 0.075] = 0
    
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
    
    # Crop out bright band on left, if present. Use the normalized image
    # instead of the thresholded image in order to differentiate the bright
    # band from breast tissue. Start 20% in from the left side since some 
    # cases start with an empty border and move left to remove the border 
    # (empty or bright).
    t_high = 0.75
    t_low  = 0.2
    start_col_left = max(int(x.shape[1]*0.2), 1)
    crop_col_left  = 0
    within_limits = False
    for col in range(start_col_left, -1, -1):
        mean = x_norm[:,col].mean()
        if within_limits and mean <= t_low:
            # empty
            crop_col_left = col
            break
        if within_limits and mean >= t_high:
            # bright
            crop_col_left = col
            break
        if mean > t_low and mean < t_high:
            # Once this is true, assume breast has been found. Useful for
            # very small breasts that extend less than 20% in from left edge.
            within_limits = True
    
    # Crop out other bright bands using thresholded image. Some bright bands
    # are not fully saturated, so it's best to remove them after this 
    # thresholding. This would not work well for the left edge since some
    # breasts take the full edge. For each edge (right, top, bottom), start
    # 20% in and move out. Stop when the mean intensity switches from below
    # the threshold to above the threshold; if it starts above the threshold,
    # then it just means the breast is spanning the entire width/height at 
    # the start position.
    t_high = 0.95
    start_col_right = max(int(x.shape[1]*0.8), 1)
    start_row_top   = max(int(x.shape[0]*0.2), 1)
    start_row_bot   = max(int(x.shape[0]*0.8), 1)
    crop_col_right = x.shape[1]
    crop_row_top   = 0
    crop_row_bot   = x.shape[0]
    prev_mean_col_right  = 1
    prev_mean_row_top    = 1
    prev_mean_row_bot    = 1
    for col in range(start_col_right, x.shape[1]):
        mean = x[:,col].mean()
        if mean > t_high and prev_mean_col_right < t_high:
            crop_col_right = col+1
            break
        prev_mean_col_right = mean
    for row in range(start_row_top, -1, -1):
        mean = x[row,:].mean()
        if mean > t_high and prev_mean_row_top < t_high:
            crop_row_top = row
            break
        prev_mean_row_top = mean
    for row in range(start_row_bot, x.shape[0]):
        mean = x[row,:].mean()
        if mean > t_high and prev_mean_row_bot < t_high:
            crop_row_bot = row+1
            break
        prev_mean_row_bot = mean
    
    # Store crop indices for edges - to be used to make sure that the final 
    # crop does not include the edges.
    crop_col_left_edge  = crop_col_left
    crop_col_right_edge = crop_col_right
    crop_row_top_edge   = crop_row_top
    crop_row_bot_edge   = crop_row_bot
    
    # Flood fill breast in order to then crop background out. Start flood 
    # at pixel 20% right from the left edge, center row. Apply the flood to 
    # a the image with the edges cropped out in order to avoid flooding the
    # edges and any artefacts that overlap the edges.
    x_view = x[crop_row_top:crop_row_bot,crop_col_left:crop_col_right]
    flood_row = max(x.shape[0]//2-crop_row_top, 0)
    flood_col = max(int(x.shape[1]*0.2)-crop_col_left, 0)
    x_fill = binary_opening(x_view, selem=np.ones((5,5)))
    x_fill = binary_closing(x_fill, selem=np.ones((5,5)))
    m_view = flood(x_fill, (flood_row, flood_col), connectivity=1)
    m = np.zeros(x.shape, dtype=np.bool)
    m[crop_row_top:crop_row_bot,crop_col_left:crop_col_right] = m_view
    
    # Crop out background outside of breast. Start 20% in from the left side
    # at the center row and move outward until the columns or rows are empty.
    # For every row or column mean, ignore 10% of each end of the vector in 
    # order to avoid problematic edges. Note that the mask already has edges
    # cropped out, if needed, but this may be imperfect.
    flood_row_adjusted = flood_row+crop_row_top
    flood_col_adjusted = flood_col+crop_col_left
    frac_row = int(m.shape[0]*0.1)
    frac_col = int(m.shape[1]*0.1)
    for col in range(flood_col_adjusted, m.shape[1]):
        if not np.any(m[frac_row:-frac_row,col]):
            crop_col_right = min(crop_col_right, col+1)
            break
    for row in range(flood_row_adjusted, -1, -1):
        if not np.any(m[row,frac_col:-frac_col]):
            crop_row_top = max(crop_row_top, row)
            break
    for row in range(flood_row_adjusted, m.shape[0]):
        if not np.any(m[row,frac_col:-frac_col]):
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
    
    # Expand crop 5% to the right and 10% up and down in order to avoid 
    # clipping any breast.
    crop_col_right = min(int(crop_col_right+x.shape[1]*0.05),
                         crop_col_right_edge)
    crop_row_top   = max(int(crop_row_top-x.shape[0]*0.1),
                         crop_row_top_edge)
    crop_row_bot   = min(int(crop_row_bot+x.shape[0]*0.1),
                         crop_row_bot_edge)
    
    # Apply crop.
    image = image[crop_row_top:crop_row_bot,crop_col_left:crop_col_right]
    if mask is not None:
        mask = mask[crop_row_top:crop_row_bot,crop_col_left:crop_col_right]
        return image, mask
    
    return image


if __name__ == '__main__':
    args = get_args()
    create_dataset(args)