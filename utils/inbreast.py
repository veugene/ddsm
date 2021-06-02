from collections import OrderedDict

import h5py
import numpy as np

from data_tools.data_augmentation import image_random_transform
from data_tools.wrap import multi_source_array

 
def prepare_data_inbreast(path, fold, masked_fraction=0,
                          drop_masked=False, split_seed=0, rng=None):
    """
    Convenience function to prepare INbreast data split into training and
    validation subsets.
    
    path (string) : Path of the h5py file containing the INbreast data.
    fold (int) : The fold in {0,1,2,3} for 4-fold cross-validation.
    masked_fraction (float) : The fraction in [0, 1.] of cases in the 
        training set for which  to return segmentation masks as None
    drop_masked (bool) : Whether to omit cases with "masked" segmentations.
    split_seed (int) : The random seed used to determine the data split.
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
        f = h5py.File(path, mode='r')
    except:
        print("Failed to open data: {}".format(path))
        raise
    
    # 25% of the data for test, 10% for validation. The validation
    # subset is randomly sampled from the data that isn't in the test
    # fold.
    cases_s = [c for c in f.keys() if 's' in f[c]]
    cases_h = [c for c in f.keys() if 'h' in f[c]]
    n_cases = len(cases_s)
    rng_split = np.random.RandomState(split_seed)
    rng_split.shuffle(cases_s)
    assert fold in [0, 1, 2, 3]
    indices = [int(x*n_cases) for x in [0.25, 0.5, 0.75, 1]]
    fold_cases = {0: cases_s[0         :indices[0]],
                  1: cases_s[indices[0]:indices[1]],
                  2: cases_s[indices[1]:indices[2]],
                  3: cases_s[indices[2]:indices[3]]}
    test = fold_cases[fold]
    not_test = sum([fold_cases[x] for x in {0,1,2,3}-{fold}], [])
    rng_valid = np.random.RandomState(fold)     # Different for each fold.
    idx_valid = rng_valid.permutation(len(not_test))[:int(0.1*n_cases)]
    split = {'train': [x for i, x in enumerate(not_test)
                       if i not in idx_valid],
             'valid': [x for i, x in enumerate(not_test)
                       if i in idx_valid],
             'test' : test}
    
    # `masked_fraction` cases will either be dropped from the training
    # set or have their segmentations set to None.
    n_train = len(split['train'])
    n_masked = int(min(n_train*masked_fraction+0.5, n_train))
    masked_indices = rng.permutation(n_train)[:n_masked]
    n_unlabeled = sum([sum(len(f[c]['s']) for c in split['train'][idx])
                      for idx in masked_indices])
    n_total   = sum([len(f[c]['s']) for c in split['train']])
    print("DEBUG: A total of {}/{} images are labeled."
          "".format(n_total-n_unlabeled, n_total))
    
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
    data['train']['h'] = [f[c]['h'][x] for c in cases_h
                          if c not in split['valid']+split['test']
                          for x in f[c]['h']]
    data['train']['s'] = []
    data['train']['m'] = []
    data['valid']['h'] = data['train']['h']    # HACK
    data['valid']['s'] = [f[c]['s'][x] for c in split['valid']
                          for x in f[c]['s']]
    data['valid']['m'] = [f[c]['m'][x] for c in split['valid']
                          for x in f[c]['m']]
    data['test']['h']  = data['train']['h']    # HACK
    data['test']['s']  = [f[c]['s'][x] for c in split['test']
                          for x in f[c]['s']]
    data['test']['m']  = [f[c]['s'][x] for c in split['test']
                          for x in f[c]['m']]
    for i in range(n_train):
        case = split['train'][i]
        if i in masked_indices:
            # Mask out or drop.
            if drop_masked:
                continue    # Drop.
            n_images = len(f[case]['m'])
            data['train']['m'].append([None]*n_images)
        else:
            # Keep.
            data['train']['m'].extend([f[case]['m'][x] for x in f[case]['m']])
        data['train']['s'].extend([f[case]['s'][x] for x in f[case]['s']])
    
    # Merge all arrays in each list of arrays.
    for key in data.keys():
        # HACK: we may have a situation where the number of sick examples
        # is greater than the number of healthy. In that case, we should
        # duplicate the healthy set M times so that it has a bigger size
        # than the sick set.
        m = 1
        len_h = sum([len(elem) for elem in data[key]['h']])
        len_s = sum([len(elem) for elem in data[key]['s']])
        if len_h < len_s:
            m = int(np.ceil(len_s / len_h))
        data[key]['h'] = multi_source_array(data[key]['h']*m)
        data[key]['s'] = multi_source_array(data[key]['s'])
        data[key]['m'] = multi_source_array(data[key]['m'])
    return data


def preprocessor_inbreast(crop_to=None, data_augmentation_kwargs=None):
    """
    Preprocessor function to pass to a data_flow, for INbreast data.
    
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
