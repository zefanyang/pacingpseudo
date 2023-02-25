#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 11/19/2021 10:39 PM
# @Author: yzf
import argparse
from glob import glob
import os
import cv2
import time
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

# Median spacing over cases
NUM_CLASSES = 5  # include background
IGNORED_INDEX = 5
TARGET_SPACING = (1.62, 1.62)
TARGET_HEIGHT, TARGET_WIDTH = (256, 256)

CLASS_MAPPING = {
    'background': 0,
    'liver': 1,
    'right_kidney': 2,
    'left_kidney': 3,
    'spleen': 4,
    'ignored_region': 5
}

def dicom_to_nifti():
    def _read_dicoms(case_path):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(case_path)
        reader.SetFileNames(dicom_names)
        image_sitk = reader.Execute()
        image_arr = sitk.GetArrayFromImage(image_sitk)  # z, y, x
        origin = image_sitk.GetOrigin()
        spacing = image_sitk.GetSpacing()
        direction = image_sitk.GetDirection()
        return image_sitk, dicom_names, image_arr, origin, spacing, direction

    def _make_gt_nifti(ground_path, dicom_names, origin, spacing, direction):
        img_ls = []
        for nam in dicom_names:
            nam = nam.rstrip('.dcm') + '.png'
            img_ls.append(np.asarray(Image.open(os.path.join(ground_path, nam))))
        img_arr = np.stack(img_ls, axis=0)  # z, y, x
        img_arr = img_arr / 63
        img_arr = img_arr.astype(np.uint8)
        img_sitk = sitk.GetImageFromArray(img_arr)
        img_sitk.SetOrigin(origin)
        img_sitk.SetSpacing(spacing)
        img_sitk.SetDirection(direction)
        return img_sitk

    raw_data_path = './raw_data/Train_Sets/MR'
    nifti_data_path = './nifti_data'
    os.makedirs(nifti_data_path, exist_ok=True)
    patient_path_ls = list(glob.glob(os.path.join(raw_data_path, '*')))
    for pat_path in patient_path_ls:
        pat_id = pat_path.rpartition('/')[-1]
        pat_out_path = os.path.join(nifti_data_path, pat_id)
        os.makedirs(pat_out_path, exist_ok=True)
        # Image
        t1_inphase_path = os.path.join(pat_path, 'T1DUAL/DICOM_anon/InPhase')
        t1_outphase_path = os.path.join(pat_path, 'T1DUAL/DICOM_anon/OutPhase')

        t1_inphase_sitk, dicom_names, _, origin1, spacing1, direction1 = _read_dicoms(t1_inphase_path)
        t1_outphase_sitk, _, _, origin2, spacing2, direction2 = _read_dicoms(t1_outphase_path)
        assert origin1 == origin2 and spacing1 == spacing2 and direction1 == direction2
        sitk.WriteImage(t1_inphase_sitk, os.path.join(pat_out_path, f'patient{pat_id}_t1_inphase_img.nii.gz'))
        sitk.WriteImage(t1_outphase_sitk, os.path.join(pat_out_path, f'patient{pat_id}_t1_outphase_img.nii.gz'))

        # Ground truth
        t1_ground_path = os.path.join(pat_path, 'T1DUAL/Ground')
        dicom_names = [p.rpartition('/')[-1] for p in dicom_names]
        ground_sitk = _make_gt_nifti(t1_ground_path, dicom_names, origin1, spacing1, direction1)
        sitk.WriteImage(ground_sitk, os.path.join(pat_out_path, f'patient{pat_id}_t1_gt.nii.gz'))

    for pat_path in patient_path_ls:
        pat_id = pat_path.rpartition('/')[-1]
        pat_out_path = os.path.join(nifti_data_path, pat_id)
        # Image
        t2_spir_path = os.path.join(pat_path, 'T2SPIR/DICOM_anon')
        t2_spir_sitk, dicom_names, _, origin, spacing, direction = _read_dicoms(t2_spir_path)
        sitk.WriteImage(t2_spir_sitk, os.path.join(pat_out_path, f'patient{pat_id}_t2_spir_img.nii.gz'))

        # Ground truth
        t2_ground_path = os.path.join(pat_path, 'T2SPIR/Ground')
        dicom_names = [p.rpartition('/')[-1] for p in dicom_names]
        ground_sitk = _make_gt_nifti(t2_ground_path, dicom_names, origin, spacing, direction)
        sitk.WriteImage(ground_sitk, os.path.join(pat_out_path, f'patient{pat_id}_t2_gt.nii.gz'))
    return

def preprocess():
    def _check_and_clip(img):
        assert not np.any(np.isnan(img))
        lower_limit = np.percentile(img, 0.5)
        upper_limit = np.percentile(img, 99.5)
        img = np.clip(img, a_min=lower_limit, a_max=upper_limit)
        return img

    def _get_affine_params(data_sitk):
        origin = data_sitk.GetOrigin()
        spacing = data_sitk.GetSpacing()
        direction = data_sitk.GetDirection()
        return origin, spacing, direction

    def _scribble_index_mapping(data):
        mask1 = data == 0  # ignored region
        mask2 = data == 5  # background
        data[mask1] = 5
        data[mask2] = 0
        return data

    def _resize_image(img, new_size, interpolation=cv2.INTER_CUBIC):
        z, _, _ = img.shape
        out = []
        for k in range(z):
            tmp = cv2.resize(img[k], new_size, interpolation=interpolation)
            out.append(tmp)
        return np.array(out)

    def _resize_label(img, new_size, no_classes, interpolation):
        def _to_one_hot(arr, no_classes):
            h, w = arr.shape
            arr_one_hot = np.zeros((no_classes, h, w), dtype=np.float32)
            for cls in range(no_classes):
                arr_one_hot[cls][arr == cls] = 1
            return arr_one_hot

        z, h, w = img.shape
        out = []
        for k in range(z):
            tmp_one_hot = _to_one_hot(img[k], no_classes)  # one-hot encoding label of slice
            tmp_out = np.zeros((no_classes, new_size[1], new_size[0]))  # container of channel-wise interpolation output
            for c in range(no_classes):
                tmp_out[c, :] = cv2.resize(tmp_one_hot[c], new_size, interpolation=interpolation)  # store
            tmp_out = np.argmax(tmp_out, axis=0)
            out.append(tmp_out)
        return np.array(out)

    def _pad_and_crop(img, new_size, fill=0):
        z, y, x = img.shape
        assert any(i % 2 for i in new_size) == False  # divisible by 2

        pad_z = (0, 0)
        pad_x = (int(np.ceil(max(0, TARGET_WIDTH - x) / 2)), int(np.floor(max(0, TARGET_WIDTH - x) / 2)))
        pad_y = (int(np.ceil(max(0, TARGET_HEIGHT - y) / 2)), int(np.floor(max(0, TARGET_HEIGHT - y) / 2)))
        img = np.pad(img, (pad_z, pad_y, pad_x), mode='constant', constant_values=fill)

        z_, y_, x_ = img.shape
        delta_x = new_size[0] // 2
        delta_y = new_size[1] // 2
        x0 = x_ // 2
        y0 = y_ // 2

        out = []
        for k in range(z_):
            out.append(img[k, y0 - delta_y: y0 + delta_y, x0 - delta_x: x0 + delta_x])
        return np.array(out)

    def _preprocessing_pipeline(img_arr, spacing, mode='img'):
        assert mode in ['img', 'lab', 'scb']
        _, h, w = img_arr.shape
        dy, dx, _ = spacing
        dy_target, dx_target = TARGET_SPACING

        # resample
        scale_y = dy / dy_target
        scale_x = dx / dx_target
        new_h = int(scale_y * h)
        new_w = int(scale_x * w)

        if mode == 'img':
            interpolation_mode = cv2.INTER_CUBIC
            img_arr = _check_and_clip(img_arr)
            img_arr = _resize_image(img_arr, new_size=(new_w, new_h), interpolation=interpolation_mode)
        elif mode == 'lab':
            interpolation_mode = cv2.INTER_LINEAR
            img_arr = _resize_label(img_arr, new_size=(new_w, new_h), no_classes=NUM_CLASSES,
                                    interpolation=interpolation_mode)
        elif mode == 'scb':
            interpolation_mode = cv2.INTER_LINEAR
            img_arr = _resize_label(img_arr, new_size=(new_w, new_h), no_classes=NUM_CLASSES + 1,
                                    interpolation=interpolation_mode)

        # pad or crop
        if mode == 'img' or mode == 'lab':
            fill = 0
        elif mode == 'scb':
            fill = IGNORED_INDEX
        img_arr = _pad_and_crop(img_arr, new_size=(TARGET_WIDTH, TARGET_HEIGHT), fill=fill)
        return img_arr

    def _data_sanity_check(img, lab, scb):
        assert not np.any(np.isnan(img))
        assert not np.any(np.isnan(lab))
        assert not np.any(np.isnan(scb))
        for cls in np.unique(lab):
            assert cls in [0, 1, 2, 3, 4]
        for cls in np.unique(scb):
            assert cls in [0, 1, 2, 3, 4, 5]

    cases = [1, 2, 3, 5, 8, 10, 13, 15, 19, 20, 21, 22, 31, 32, 33, 34, 36, 37, 38, 39]
    nifti_data_path = './raw_data/nifti_data'
    scb_data_path = './raw_data/scribble_data'
    preprocessed_data_path = './preprocessed_data'
    spacing_ls = []
    shape_ls = []

    # T1
    for idx in cases:
        tic = time.time()
        # File
        t1_ip_img_file = os.path.join(nifti_data_path, str(idx), f'patient{idx}_t1_inphase_img.nii.gz')
        t1_op_img_file = os.path.join(nifti_data_path, str(idx), f'patient{idx}_t1_outphase_img.nii.gz')
        t1_lab_file = os.path.join(nifti_data_path, str(idx), f'patient{idx}_t1_gt.nii.gz')
        t1_scb_file = os.path.join(scb_data_path, str(idx), f'patient{idx}_t1_scb.nii.gz')

        # Read image
        t1_ip_sitk = sitk.ReadImage(t1_ip_img_file)
        t1_op_sitk = sitk.ReadImage(t1_op_img_file)
        t1_lab_sitk = sitk.ReadImage(t1_lab_file)
        t1_scb_sitk = sitk.ReadImage(t1_scb_file)
        t1_ip_img = sitk.GetArrayFromImage(t1_ip_sitk)  # z, y, x
        t1_op_img = sitk.GetArrayFromImage(t1_op_sitk)
        t1_lab = sitk.GetArrayFromImage(t1_lab_sitk)
        t1_scb = sitk.GetArrayFromImage(t1_scb_sitk)
        origin, spacing, direction = _get_affine_params(t1_ip_sitk)
        print("T1, case: {:2d}, spacing: {:4f} {:4f} {:4f}, shape: {} {} {}".format(int(idx), *spacing, *t1_ip_img.shape), end=', ')
        spacing_ls.append(list(spacing))
        shape_ls.append(list(t1_ip_img.shape))

        # Data sanity check
        _data_sanity_check(t1_ip_img, t1_lab, t1_scb)
        _data_sanity_check(t1_op_img, t1_lab, t1_scb)

        # Preprocess
        t1_ip_img = _preprocessing_pipeline(t1_ip_img, spacing, mode='img')
        t1_op_img = _preprocessing_pipeline(t1_op_img, spacing, mode='img')
        t1_lab = _preprocessing_pipeline(t1_lab, spacing, mode='lab')
        t1_scb = _scribble_index_mapping(t1_scb)
        t1_scb = _preprocessing_pipeline(t1_scb, spacing, mode='scb')

        # Save image
        fd = os.path.join(preprocessed_data_path, str(idx))
        os.makedirs(fd, exist_ok=True)
        np.savez(os.path.join(fd, f'patient{idx}_t1_inphase'), img=t1_ip_img, lab=t1_lab, scb=t1_scb)
        np.savez(os.path.join(fd, f'patient{idx}_t1_outphase'), img=t1_op_img, lab=t1_lab, scb=t1_scb)
        toc = time.time()
        print('{:4f} s/case'.format(toc - tic))

    # T2
    for idx in cases:
        tic = time.time()
        # File
        t2_img_file = os.path.join(nifti_data_path, str(idx), f'patient{idx}_t2_spir_img.nii.gz')
        t2_lab_file = os.path.join(nifti_data_path, str(idx), f'patient{idx}_t2_gt.nii.gz')
        t2_scb_file = os.path.join(scb_data_path, str(idx), f'patient{idx}_t2_scb.nii.gz')

        # Read image
        t2_sitk = sitk.ReadImage(t2_img_file)
        t2_lab_sitk = sitk.ReadImage(t2_lab_file)
        t2_scb_sitk = sitk.ReadImage(t2_scb_file)
        t2_img = sitk.GetArrayFromImage(t2_sitk)
        t2_lab = sitk.GetArrayFromImage(t2_lab_sitk)
        t2_scb = sitk.GetArrayFromImage(t2_scb_sitk)
        origin, spacing, direction = _get_affine_params(t2_sitk)
        print("T2, case: {:2d}, spacing: {:4f} {:4f} {:4f}, shape: {} {} {}".format(int(idx), *spacing, *t1_ip_img.shape), end=', ')
        spacing_ls.append(list(spacing))
        shape_ls.append(list(t2_img.shape))

        # Data sanity check
        _data_sanity_check(t2_img, t2_lab, t2_scb)

        # Preprocess
        t2_img = _preprocessing_pipeline(t2_img, spacing, mode='img')
        t2_lab = _preprocessing_pipeline(t2_lab, spacing, mode='lab')
        t2_scb = _scribble_index_mapping(t2_scb)
        t2_scb = _preprocessing_pipeline(t2_scb, spacing, mode='scb')

        # Save image
        fd = os.path.join(preprocessed_data_path, str(idx))
        os.makedirs(fd, exist_ok=True)
        np.savez(os.path.join(fd, f'patient{idx}_t2'), img=t2_img, lab=t2_lab, scb=t2_scb)

        toc = time.time()
        print('{:4f} s/case'.format(toc - tic))

    # Median spacing and shape
    spacing_arr = np.array(spacing_ls)
    shape_arr = np.array(shape_ls)
    median_spacing = np.percentile(spacing_arr, 50, 0, interpolation='nearest')
    print("T2 Median spacing at {:4f}, {:4f}, {:4f}".format(*median_spacing))
    new_shape_arr = shape_arr[:, ::-1] * (spacing_arr / median_spacing)
    median_shape = np.percentile(new_shape_arr, 50, 0, interpolation='nearest')
    print("T2 Median shape at target spacing {}, {}, {}".format(*median_shape))

def save_as_slices():
    def _save_as_slices(file, out_fd):
        os.makedirs(out_fd, exist_ok=True)
        _, _, case_id, case_name = file.split('/')
        case_name = case_name.split('.')[0]
        data = np.load(file, allow_pickle=True)
        img = data.get('img')
        lab = data.get('lab')
        scb = data.get('scb')
        d, _, _ = img.shape
        for i in range(d):
            img_2d = img[i]
            lab_2d = lab[i]
            scb_2d = None
            if scb.shape:
                scb_2d = scb[i]
            np.savez(os.path.join(out_fd, case_name+f'_{i}'), uid=case_name+f'_{i}', img=img_2d, lab=lab_2d, scb=scb_2d)
        return

    preprocessed_data_path = './preprocessed_data'
    model_input_t1_path = './train_test_split/data_2d/t1'
    model_input_t2_path = './train_test_split/data_2d/t2'
    cases = [1, 2, 3, 5, 8, 10, 13, 15, 19, 20, 21, 22, 31, 32, 33, 34, 36, 37, 38, 39]

    # Training set
    for case_id in tqdm(cases):
        t1_ip_file = os.path.join(preprocessed_data_path, str(case_id), f'patient{case_id}_t1_inphase.npz')
        t1_op_file = os.path.join(preprocessed_data_path, str(case_id), f'patient{case_id}_t1_outphase.npz')
        t2_file = os.path.join(preprocessed_data_path, str(case_id), f'patient{case_id}_t2.npz')
        _save_as_slices(t1_ip_file, os.path.join(model_input_t1_path, str(case_id)))
        _save_as_slices(t1_op_file, os.path.join(model_input_t1_path, str(case_id)))
        _save_as_slices(t2_file, os.path.join(model_input_t2_path, str(case_id)))
    return

def five_fold_split():
    t1_slice_path = './train_test_split/data_2d/t1'
    t2_slice_path = './train_test_split/data_2d/t2'
    t1_txt_path = './train_test_split/five_fold_split/t1'
    t2_txt_path = './train_test_split/five_fold_split/t2'
    os.makedirs(t1_txt_path, exist_ok=True)
    os.makedirs(t2_txt_path, exist_ok=True)

    np.random.seed(1)
    n = 5
    cases = [1, 2, 3, 5, 8, 10, 13, 15, 19, 20, 21, 22, 31, 32, 33, 34, 36, 37, 38, 39]
    np.random.shuffle(cases)
    assert len(set(cases)) == len(cases)
    folds = [list(_) for _ in np.array_split(cases, n)]

    for i in range(n):
        train, test = [], []
        for j in range(n):
            if j == i:
                test += list(folds[j])
            else:
                train += list(folds[j])
        assert set(train + test) == set(sum(folds, []))
        print(f'Fold {i}')
        print('Train: {}'.format(train))
        print('Test: {}'.format(test))

        # T1
        train_files = []
        for pat in train:
            pat = os.path.join(t1_slice_path, str(pat))
            files = [_.lstrip('./')+'\n' for _ in glob(os.path.join(pat, '*.npz'))]
            train_files += files
        with open(os.path.join(t1_txt_path, f'train_fold{i}.txt'), 'w') as f:
            f.writelines(train_files)
        print("Number of T1 train slices {}".format(len(train_files)))

        test_files = []
        for pat in test:
            pat = os.path.join(t1_slice_path, str(pat))
            files = [_.lstrip('./')+'\n' for _ in glob(os.path.join(pat, '*.npz'))]
            test_files += files
        with open(os.path.join(t1_txt_path, f'test_fold{i}.txt'), 'w') as f:
            f.writelines(test_files)
        print("Number of T1 test slices {}".format(len(test_files)))

        # T2
        train_files = []
        for pat in train:
            pat = os.path.join(t2_slice_path, str(pat))
            files = [_.lstrip('./')+'\n' for _ in glob(os.path.join(pat, '*.npz'))]
            train_files += files
        with open(os.path.join(t2_txt_path, f'train_fold{i}.txt'), 'w') as f:
            f.writelines(train_files)
        print("Number of T2 train slices {}".format(len(train_files)))

        test_files = []
        for pat in test:
            pat = os.path.join(t2_slice_path, str(pat))
            files = [_.lstrip('./')+'\n' for _ in glob(os.path.join(pat, '*.npz'))]
            test_files += files
        with open(os.path.join(t2_txt_path, f'test_fold{i}.txt'), 'w') as f:
            f.writelines(test_files)
        print("Number of T2 test slices {}".format(len(test_files)))
    return

def visual_sanity_check():
    import matplotlib as mpl
    cmap = mpl.colors.ListedColormap(['whitesmoke', 'red', 'limegreen', 'blue', 'gold', 'black'])
    norm = mpl.colors.BoundaryNorm(boundaries=[0, 1, 2, 3, 4, 5, 6], ncolors=cmap.N)  # Assume that intervals are (half-open) right-open, like [0, 1)。

    preprocessed_data_path = './preprocessed_data'
    visualized_data_path = './visualized_data'
    cases = [1, 2, 3, 5, 8, 10, 13, 15, 19, 20, 21, 22, 31, 32, 33, 34, 36, 37, 38, 39]

    for case_id in tqdm(cases):
        t1_ip_file = os.path.join(preprocessed_data_path, str(case_id), f'patient{case_id}_t1_inphase.npz')
        t1_op_file = os.path.join(preprocessed_data_path, str(case_id), f'patient{case_id}_t1_outphase.npz')
        t1_ip_data = np.load(t1_ip_file, allow_pickle=True)
        t1_op_data = np.load(t1_op_file, allow_pickle=True)
        t1_ip_img = t1_ip_data.get('img')
        t1_op_img = t1_op_data.get('img')
        t1_lab = t1_ip_data.get('lab')
        t1_scb = t1_ip_data.get('scb')

        # Polish the figure
        d, h, w = t1_ip_img.shape
        for i in range(d):
            t1_ip_2d = t1_ip_img[i]
            t1_op_2d = t1_op_img[i]
            lab_2d = t1_lab[i]
            fig = plt.figure(figsize=(8, 4), dpi=200)
            ax = plt.subplot(1, 4, 1)
            if t1_scb.shape:
                scb_2d = t1_scb[i]
                # mask = scb_2d == NUM_CLASSES
                # scb_2d[mask] = 0
            ax.imshow(t1_ip_2d, 'gray')
            ax.set_axis_off(); ax.set_title('T1 inphase')
            ax = plt.subplot(1, 4, 2)
            ax.imshow(t1_op_2d, 'gray')
            ax.set_axis_off(); ax.set_title('T1 outphase')
            ax = plt.subplot(1, 4, 3)
            ax.imshow(lab_2d, norm=norm, cmap=cmap, interpolation='nearest')  # The spectral colormaps shall be well fitted.
            ax.set_axis_off(); ax.set_title('Label')
            if t1_scb.shape:
                ax = plt.subplot(1, 4, 4)
                ax.imshow(scb_2d, norm=norm, cmap=cmap, interpolation='nearest')
                ax.set_axis_off(); ax.set_title('Scribble')
            plt.tight_layout()
            plt.savefig(os.path.join(visualized_data_path, f'patient{case_id}_t1_{i}.jpg'), dpi=300)
            # plt.show()
            plt.close()

    for case_id in tqdm(cases):
        t2_file = os.path.join(preprocessed_data_path, str(case_id), f'patient{case_id}_t2.npz')
        t2_data = np.load(t2_file, allow_pickle=True)
        t2_img = t2_data.get('img')
        t2_lab = t2_data.get('lab')
        t2_scb = t2_data.get('scb')

        # Polish the figure
        d, h, w = t2_img.shape
        for i in range(d):
            t2_img_2d = t2_img[i]
            lab_2d = t2_lab[i]
            fig = plt.figure(figsize=(8, 4), dpi=200)
            ax = plt.subplot(1, 4, 1)
            if t2_scb.shape:
                scb_2d = t2_scb[i]
            ax.imshow(t2_img_2d, 'gray')
            ax.set_axis_off(); ax.set_title('T2')
            ax = plt.subplot(1, 4, 2)
            ax.imshow(lab_2d, norm=norm, cmap=cmap, interpolation='nearest')
            ax.set_axis_off(); ax.set_title('Label')
            if t2_scb.shape:
                ax = plt.subplot(1, 4, 3)
                ax.imshow(scb_2d, norm=norm, cmap=cmap, interpolation='nearest')
                ax.set_axis_off(); ax.set_title('Scribble')
            plt.tight_layout()
            plt.savefig(os.path.join(visualized_data_path, f'patient{case_id}_t2_{i}.jpg'), dpi=300)
            # plt.show()
            plt.close()

parser = argparse.ArgumentParser()
parser.add_argument('--procedure',
                    type=str,
                    default='split_data',
                    choices=['dicom_to_nifti', 'preprocess', 'save_as_slices', 'split_data', 'visual_sanity_check'], )
parser.add_argument('--split_mode',
                    type=str,
                    default='five_fold_split')

if __name__ == '__main__':
    args = parser.parse_args()
    print(args.procedure)
    if args.procedure == 'dicom_to_nifti':
        dicom_to_nifti()
    elif args.procedure == 'preprocess':
        preprocess()
    elif args.procedure == 'save_as_slices':
        save_as_slices()
    elif args.procedure == 'split_data':
        if args.split_mode == 'five_fold_split':
            five_fold_split()
    elif args.procedure == 'visual_sanity_check':
        visual_sanity_check()