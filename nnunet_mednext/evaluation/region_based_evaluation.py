from copy import deepcopy
from multiprocessing.pool import Pool

from batchgenerators.utilities.file_and_folder_operations import *
from medpy import metric
import SimpleITK as sitk
import numpy as np
from nnunet_mednext.configuration import default_num_threads
from nnunet_mednext.postprocessing.consolidate_postprocessing import collect_cv_niftis
from surface_distance import metrics

def get_brats_regions():
    """
    this is only valid for the brats data in here where the labels are 1, 2, and 3. 
    The original brats data have a different labeling convention!
    :return:
    """
    regions = {
        "whole tumor": (1, 2, 3),
        "tumor core": (2, 3),
        "enhancing tumor": (3,)
    }
    return regions


def get_KiTS_regions():
    regions = {
        "kidney incl tumor": (1, 2),
        "tumor": (2,)
    }
    return regions


def get_kits19_regions():
    regions = {
        "kidney": (1,),
        "tumor": (2,)
    }
    return regions


def get_amos22_regions():
    regions = {
        "spleen": (1,),
        "right kidney": (2,),
        "left kidney": (3,),
        "gall bladder": (4,),
        "esophagus": (5,),
        "liver": (6,),
        "stomach": (7,),
        "aorta": (8,),
        "postcava": (9,),
        "pancreas": (10,),
        "right adrenal gland": (11,),
        "left adrenal gland": (12,),
        "duodenum": (13,),
        "bladder": (14,),
        "prostrate/uterus": (15,),
    }
    return regions


def get_MultiACT_regions():
    regions = {
        "spleen": (1,),
        "left kidney": (2,),
        "gallbladder": (3,),
        "esophagus": (4,),
        "liver": (5,),
        "stomach": (6,),
        "pancreas": (7,),
        "duodenum": (8,),
    }
    return regions


def get_btcv_regions():
    regions = {
        "spleen": (1,),
        "right kidney": (2,),
        "left kidney": (3,),
        "gallbladder": (4,),
        "esophagus": (5,),
        "liver": (6,),
        "stomach": (7,),
        "aorta": (8,),
        "inferior vena cava": (9,),
        "portal and splenic vein": (10,),
        "pancreas": (11,),
        "right adrenal gland": (12,),
        "left adrenal gland": (13,),
    }
    return regions

def create_region_from_mask(mask, join_labels: tuple):
    mask_new = np.zeros_like(mask, dtype=np.uint8)
    for l in join_labels:
        mask_new[mask == l] = 1
    return mask_new

def evaluate_case_dc(file_pred: str, file_gt: str, regions):
    image_gt = sitk.GetArrayFromImage(sitk.ReadImage(file_gt))
    image_pred = sitk.GetArrayFromImage(sitk.ReadImage(file_pred))
    results = []
    for r in regions:
        mask_pred = create_region_from_mask(image_pred, r)
        mask_gt = create_region_from_mask(image_gt, r)
        dc = np.nan if np.sum(mask_gt) == 0 and np.sum(mask_pred) == 0 else metric.dc(mask_pred, mask_gt)
        results.append(dc)
    return results

def evaluate_case_sdc(file_pred: str, file_gt: str, regions):
    image_gt = sitk.GetArrayFromImage(sitk.ReadImage(file_gt))
    image_pred = sitk.GetArrayFromImage(sitk.ReadImage(file_pred))
    results = []
    for r in regions:
        mask_pred = create_region_from_mask(image_pred, r).astype(bool)
        mask_gt = create_region_from_mask(image_gt, r).astype(bool)
        if np.sum(mask_gt) == 0 and np.sum(mask_pred) == 0:
            sdc = np.nan  
        else:
            surface_distances = metrics.compute_surface_distances(
                                        mask_gt, mask_pred, [1,1,1]
                                    )
            sdc =  metrics.compute_surface_dice_at_tolerance(
                                        surface_distances, 
                                        tolerance_mm=1)
        results.append(sdc)
    return results

def evaluate_regions(folder_predicted: str, folder_gt: str, regions: dict, processes=default_num_threads,
                    measures=['dc', 'surface_dc']):
    
    measure_funcs = {
        'surface_dc': evaluate_case_sdc,
        'dc': evaluate_case_dc
    }

    for m in measures:
        region_names = list(regions.keys())
        files_in_pred = subfiles(folder_predicted, suffix='.nii.gz', join=False)
        files_in_gt = subfiles(folder_gt, suffix='.nii.gz', join=False)
        have_no_gt = [i for i in files_in_pred if i not in files_in_gt]
        assert len(have_no_gt) == 0, "Some files in folder_predicted have not ground truth in folder_gt"
        have_no_pred = [i for i in files_in_gt if i not in files_in_pred]
        if len(have_no_pred) > 0:
            print("WARNING! Some files in folder_gt were not predicted (not present in folder_predicted)!")

        files_in_gt.sort()
        files_in_pred.sort()

        # run for all cases
        full_filenames_gt = [join(folder_gt, i) for i in files_in_pred]
        full_filenames_pred = [join(folder_predicted, i) for i in files_in_pred]

        p = Pool(processes)
        res = p.starmap(measure_funcs[m], zip(full_filenames_pred, full_filenames_gt, [list(regions.values())] * len(files_in_gt)))
        p.close()
        p.join()

        all_results = {r: [] for r in region_names}
        with open(join(folder_predicted, f'summary_{m}.csv'), 'w') as f:
            f.write("casename")
            for r in region_names:
                f.write(",%s" % r)
            f.write("\n")
            for i in range(len(files_in_pred)):
                f.write(files_in_pred[i][:-7])
                result_here = res[i]
                for k, r in enumerate(region_names):
                    dc = result_here[k]
                    f.write(",%02.4f" % dc)
                    all_results[r].append(dc)
                f.write("\n")

            f.write('mean')
            for r in region_names:
                f.write(",%02.4f" % np.nanmean(all_results[r]))
            f.write("\n")
            f.write('median')
            for r in region_names:
                f.write(",%02.4f" % np.nanmedian(all_results[r]))
            f.write("\n")

            f.write('mean (nan is 1)')
            for r in region_names:
                tmp = np.array(all_results[r])
                tmp[np.isnan(tmp)] = 1
                f.write(",%02.4f" % np.mean(tmp))
            f.write("\n")
            f.write('median (nan is 1)')
            for r in region_names:
                tmp = np.array(all_results[r])
                tmp[np.isnan(tmp)] = 1
                f.write(",%02.4f" % np.median(tmp))
            f.write("\n")

def main():
    import argparse
    args = argparse.ArgumentParser(description='Input Argument Parser for Segmentation Net')
    args.add_argument('--folder_predicted', type=str)
    args.add_argument('--folder_gt_niftis', type=str)
    args.add_argument('--measures', nargs='+', type=str, 
                        default=['dc', 'surface_dc'])
    args.add_argument('--regions', type=str, choices=['brats21', 'kits19', 'btcv', 'amos22'])    
    args = args.parse_args()

    if args.regions == 'brats21':
        regions = get_brats_regions
    elif args.regions == 'kits19':
        regions = get_kits19_regions
    elif args.regions == 'amos22':
        regions = get_amos22_regions
    elif args.regions == 'btcv':
        regions = get_btcv_regions

    print(regions())
    evaluate_regions(
        args.folder_predicted, 
        args.folder_gt_niftis,
        regions(),
        )

# if __name__ == '__main__':
#     main()

    # collect_cv_niftis('./', './cv_niftis')
    # evaluate_regions('./cv_niftis/', './gt_niftis/', get_brats_regions())

    # import argparse
    # args = argparse.ArgumentParser(description='Input Argument Parser for Segmentation Net')
    # args.add_argument('--folder_predicted', type=str)
    # args.add_argument('--folder_gt_niftis', type=str)
    # args = args.parse_args()

    # evaluate_regions(
    #     args['folder_predicted'], 
    #     args['folder_gt_niftis'],
    #     get_brats_regions()
    #     )