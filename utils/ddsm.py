from collections import OrderedDict

import h5py
import numpy as np

from data_tools.data_augmentation import image_random_transform
from data_tools.wrap import multi_source_array


def prepare_data_ddsm(path, masked_fraction=0, drop_masked=False, rng=None):
    """
    Convenience function to prepare DDSM data split into training and
    validation subsets.
    
    path (string) : Path of the h5py file containing the DDSM data.
    masked_fraction (float) : The fraction in [0, 1.] of cases in the 
        training set for which  to return segmentation masks as None
    drop_masked (bool) : Whether to omit cases with "masked" segmentations.
    rng (numpy RandomState) : Random number generator.
    
    NOTE: The rng passed for data preparation is used to determine which 
    labels to mask out (if any); if none is passed, the default uses a
    random seed of 0.
    
    Returns dictionary: healthy slices, sick slices, and segmentations for 
    the training, validation, and testing subsets.
    """
    if rng is None:
        rng = np.random.RandomState(0)
    if masked_fraction < 0 or masked_fraction > 1:
        raise ValueError("`masked_fraction` must be in [0, 1].")
    try:
        h5py_file = h5py.File(path, mode='r')
    except:
        print("Failed to open data: {}".format(path))
        raise
    
    # Cases with these indices will either be dropped from the training
    # set or have their segmentations set to None.
    # 
    # The `masked_fraction` determines the maximal fraction of cases that
    # are to be thus removed.
    num_cases_total  = len(h5py_file['train']['m'])
    num_cases_masked = int(min(num_cases_total,
                               num_cases_total*masked_fraction+0.5))
    masked_indices = rng.permutation(num_cases_total)[:num_cases_masked]
    print("DEBUG: A total of {}/{} images are labeled."
          "".format(num_cases_total-num_cases_masked, num_cases_total))
    
    # Apply masking in one of two ways.
    # 
    # 1. Mask out the labels for cases indexed with `masked_indices` by 
    # setting the segmentation as an array of `None`.
    # 
    # OR if `drop_masked` is True:
    # 
    # 2. Remove all cases indexed with `masked_indices`.
    data = OrderedDict([('train', OrderedDict()),
                        ('valid', OrderedDict()),
                        ('test',  OrderedDict())])
    data['train']['h'] = h5py_file['train']['h']
    data['train']['s'] = []
    data['train']['m'] = []
    data['valid']['h'] = h5py_file['train']['h']    # HACK
    data['valid']['s'] = h5py_file['valid']['s']
    data['valid']['m'] = h5py_file['valid']['m']
    data['test']['h']  = h5py_file['train']['h']    # HACK
    data['test']['s']  = h5py_file['test']['s']
    data['test']['m']  = h5py_file['test']['m']
    for i in range(num_cases_total):
        if i in masked_indices:
            # Mask out or drop.
            if drop_masked:
                continue    # Drop.
            data['train']['m'].append(None)
        else:
            # Keep.
            data['train']['m'].append(h5py_file['train']['m'][i])
        data['train']['s'].append(h5py_file['train']['s'][i])
    data['train']['s'] = _list(data['train']['s'], np.uint16)
    data['train']['m'] = _list(data['train']['m'], np.uint8)
    
    # Merge all arrays in each list of arrays.
    for key in data.keys():
        # HACK: we may have a situation where the number of sick examples
        # is greater than the number of healthy. In that case, we should
        # duplicate the healthy set M times so that it has a bigger size
        # than the sick set.
        len_h = sum([len(elem) for elem in data[key]['h']])
        len_s = sum([len(elem) for elem in data[key]['s']])
        if len_h < len_s:
            m = int(np.ceil(len_s / len_h))
            data[key]['h'] = multi_source_array([data[key]['h']]*m)
    return data


def prepare_data_cbis(path, split_seed=0, rng=None, **kwargs):
    # HACK supervised dataset for debugging
    # HACK healthy is just sick, repeated
    # HACK validation set is just the test set
    # HACK kwargs is ignored
    if rng is None:
        rng = np.random.RandomState(0)
    try:
        h5py_file = h5py.File(path, mode='r')
    except:
        print("Failed to open data: {}".format(path))
        raise
    
    data = OrderedDict([('train', OrderedDict()),
                        ('valid', OrderedDict()),
                        ('test',  OrderedDict())])
    data['train']['h'] = h5py_file['train']['s']
    data['train']['s'] = h5py_file['train']['s']
    data['train']['m'] = h5py_file['train']['m']
    data['valid']['h'] = h5py_file['test']['s']
    data['valid']['s'] = h5py_file['test']['s']
    data['valid']['m'] = h5py_file['test']['m']
    data['test']['h']  = h5py_file['test']['s']
    data['test']['s']  = h5py_file['test']['s']
    data['test']['m']  = h5py_file['test']['m']
    
    return data


from torch.utils.data import Dataset
import SimpleITK as sitk
from scipy import ndimage
import os
import cv2

def get_largest_connected_component(m):
    m_labeled, n_labels = ndimage.label(m)
    sizes = ndimage.sum(m, m_labeled, index=range(1, n_labels+1))
    label = np.argmax(sizes)+1
    col, row = ndimage.find_objects(m_labeled==label)[0]
    return col, row

class png_dataset(Dataset):
    def __init__(self, image_path, mask_path, da_kwargs):
        super().__init__()
        self.image_path = image_path
        self.mask_path = mask_path
        self.da_kwargs = da_kwargs
        self._image_files = sorted(os.listdir(image_path))
    
    def __getitem__(self, idx):
        img_fn = self._image_files[idx]
        img = cv2.imread(os.path.join(self.image_path, img_fn),
                         cv2.IMREAD_GRAYSCALE)
        
        # Crop to breast.
        otsu = sitk.OtsuThresholdImageFilter()
        otsu.SetInsideValue(0)
        otsu.SetOutsideValue(1)
        img_include_sitk = otsu.Execute(sitk.GetImageFromArray(img))
        img_include = sitk.GetArrayFromImage(img_include_sitk)
        
        # Get the bounding box of the largest connected component.
        col, row = get_largest_connected_component(img_include>0)
        
        # Crop.
        #img_orig = img.copy()
        crop_x = slice(col.start, max(col.stop, col.start+256))
        crop_y = slice(row.start, max(row.stop, row.start+256))
        img = img[crop_x, crop_y]
        #print(img_fn, img_orig.shape, row, col)
        
        #img_orig = cv2.resize(src=img_orig, dsize=(256,256))
        #img_include = cv2.resize(src=img_include, dsize=(256,256))
        #from matplotlib import pyplot as plt
        #fig, ax = plt.subplots(1, 3)
        #ax[0].imshow(img_orig, cmap='gray')
        #ax[1].imshow(img_include, cmap='gray')
        #ax[2].imshow(img, cmap='gray')
        #fig.savefig(os.path.join("debug_ddsm_png",
                                 #os.path.basename(img_fn)+".png"))
        #plt.close()
        #exit()
        
        # Resize.
        img = cv2.resize(src=img, dsize=(256, 256),
                         interpolation=cv2.INTER_AREA)
        img = (img - img.min()) / (img.max() - img.min() + 1.0e-10)
        img = img[np.newaxis, :].astype(np.float32)
        fn_root = img_fn.replace("_FULL___PRE.png", "")
        mask_fn = fn_root+"_MASK___PRE.png"
        if not os.path.exists(os.path.join(self.mask_path, mask_fn)):
            mask_fn = fn_root+"_MASK_1___PRE.png"
        if not os.path.exists(os.path.join(self.mask_path, mask_fn)):
            mask_fn = fn_root+"_MASK_2___PRE.png"
        if not os.path.exists(os.path.join(self.mask_path, mask_fn)):
            mask_fn = fn_root+"_MASK_3___PRE.png"
        if not os.path.exists(os.path.join(self.mask_path, mask_fn)):
            mask_fn = fn_root+"_MASK_4___PRE.png"
        mask = cv2.imread(os.path.join(self.mask_path, mask_fn),
                          cv2.IMREAD_GRAYSCALE)
        mask = mask[crop_x, crop_y]
        mask = cv2.resize(src=mask, dsize=(256, 256),
                          interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(float)/mask.max()
        mask[mask<0.5] = 0
        mask[mask>=0.5] = 1
        mask = mask.astype(np.uint8)
        mask = mask[np.newaxis, :]
        
        # Data augmentation
        img, mask = image_random_transform(img, mask,
                                           **self.da_kwargs,
                                           n_warp_threads=1)
        img = img.copy()
        mask = mask.astype(np.uint8)
        
        return img, img, mask
    
    def __len__(self):
        return len(self._image_files)


def prepare_data_cbis_png(path, data_augmentation_kwargs):
    # HACK supervised dataset for debugging
    # HACK healthy is just sick, repeated
    # HACK validation set is just the test set
    data = {'train': png_dataset(os.path.join(path, "full", "train"),
                                 os.path.join(path, "mask", "train"),
                                 data_augmentation_kwargs),
            'valid': png_dataset(os.path.join(path, "full", "test"),
                                 os.path.join(path, "mask", "test"),
                                 data_augmentation_kwargs),
            'test' : png_dataset(os.path.join(path, "full", "test"),
                                 os.path.join(path, "mask", "test"),
                                 data_augmentation_kwargs)}
    return data


class _list(object):
    def __init__(self, indexable, dtype):
        self._items = indexable
        self.dtype = dtype
    def __getitem__(self, idx):
        elem = self._items[idx]
        if elem is not None:
            elem = elem[...]
        return elem
    def __len__(self):
        return len(self._items)


def preprocessor_ddsm(crop_to=None, data_augmentation_kwargs=None):
    """
    Preprocessor function to pass to a data_flow, for DDSM data.
    
    crop_to : An int defining the spatial size to crop all inputs to,
        after data augmentation (if any). Crops are centered. Used for
        processing images for validation.
    data_augmentation_kwargs : Dictionary of keyword arguments to pass to
        the data augmentation code (image_stack_random_transform).
    """
    def process_element(inputs):
        h, s, m = inputs

        # Float.
        h = h.astype(np.float32)
        s = s.astype(np.float32)
        
        # Expand dims.
        h = np.expand_dims(h, 0)
        s = np.expand_dims(s, 0)
        if m is not None:
            m = np.expand_dims(m, 0)
        
        # Data augmentation.
        if data_augmentation_kwargs is not None:
            h = image_random_transform(h, **data_augmentation_kwargs,
                                       n_warp_threads=1)
            _ = image_random_transform(s, m, **data_augmentation_kwargs,
                                       n_warp_threads=1)
            if m is not None:
                s, m = _
            else:
                s = _
        
        # Normalize.
        h_non_background = h>h.min()
        s_non_background = s>s.min()
        h -= h[h_non_background].mean()
        h /= 5*h[h_non_background].std()
        s -= s[s_non_background].mean()
        s /= 5*s[s_non_background].std()
        
        # Remove distant outlier intensities.
        h = np.clip(h, -1, 1)
        s = np.clip(s, -1, 1)
        
        # Change mask values from 255 to 1.
        if m is not None:
            m[m>0] = 1
        
        # Crop images (centered) -- for validation.
        if crop_to is not None:
            assert np.all(h.shape==s.shape)
            assert np.all(h.shape==m.shape)
            x, y = np.subtract(h.shape[-2:], crop_to)//2
            h = h[:, x:x+crop_to, y:y+crop_to]
            s = s[:, x:x+crop_to, y:y+crop_to]
            m = m[:, x:x+crop_to, y:y+crop_to]
                
        return h, s, m
    
    def process_batch(batch):
        elements = []
        for i in range(len(batch[0])):
            elem = process_element([b[i] for b in batch])
            elements.append(elem)
        out_batch = list(zip(*elements))
        return out_batch
    
    return process_batch
