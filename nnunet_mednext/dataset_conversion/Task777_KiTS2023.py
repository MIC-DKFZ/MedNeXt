from batchgenerators.utilities.file_and_folder_operations import *
import shutil

from nnunet_mednext.paths import nnUNet_raw_data
from nnunet_mednext.dataset_conversion.utils import generate_dataset_json

if __name__ == '__main__':
    # this is the data folder from the kits21 github repository, see https://github.com/neheller/kits21
    kits_data_dir = '/mnt/cluster-data-all/roys/raw_data/nnUNet_raw_data_base/kits23/dataset/'
    nnUNet_raw_data = '/mnt/cluster-data-all/roys/raw_data/nnUNet_raw_data_base/nnUNet_raw_data'

    # This script uses the majority voted segmentation as ground truth
    kits_segmentation_filename = 'segmentation.nii.gz'

    # Arbitrary task id. This is just to ensure each dataset ha a unique number. Set this to whatever ([0-999]) you
    # want
    task_id = 777
    task_name = "KiTS2023"

    foldername = "Task%03.0d_%s" % (task_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw_data, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    case_ids = subdirs(kits_data_dir, prefix='case_', join=False)
    for c in case_ids:
        if isfile(join(kits_data_dir, c, kits_segmentation_filename)):
            shutil.copy(join(kits_data_dir, c, kits_segmentation_filename), join(labelstr, c + '.nii.gz'))
            shutil.copy(join(kits_data_dir, c, 'imaging.nii.gz'), join(imagestr, c + '_0000.nii.gz'))

    generate_dataset_json(join(out_base, 'dataset.json'),
                          imagestr,
                          None,
                          ('CT',),
                          {
                              0: 'background',
                              1: "kidney",
                              2: "tumor",
                              3: "cyst",
                          },
                          task_name,
                          license='see https://kits-challenge.org/kits23/',
                          dataset_description='see https://kits-challenge.org/kits23/',
                          dataset_release='0')
