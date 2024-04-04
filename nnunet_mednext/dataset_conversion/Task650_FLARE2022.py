#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from collections import OrderedDict
from nnunet_mednext.paths import nnUNet_raw_data
from batchgenerators.utilities.file_and_folder_operations import *
import shutil


if __name__ == "__main__":
    base = "/mnt/cluster-data-all/roys/raw_data/nnUNet_raw_data_base/nnUNet_raw_data"

    task_id = 650
    task_name = "FLARE2022"
    prefix = 'FLARE22_Tr'

    foldername = "Task%03.0d_%s" % (task_id, task_name)

    base = join(base, foldername)
    imagestr = join(base, "imagesTr")
    imagests = join(base, "imagesTs")
    labelstr = join(base, "labelsTr")

    train_patient_names = []
    label_patient_names = []
    test_patient_names = []
    train_patients = subfiles(imagestr, join=False, suffix = 'nii.gz')
    for p in train_patients:
        print(p)
        label_name = p[0:15]+".nii.gz"
        train_patient_names.append(label_name) # nnUNet will append 0000 on it's own
        label_patient_names.append(label_name)
        print(label_name)
    #     train_patient_name = f'{prefix}_{serial_number:03d}.nii.gz'
    #     label_file = join(label_folder, f'label{p[3:]}')
    #     image_file = join(train_folder, p)
    #     shutil.copy(image_file, join(imagestr, f'{train_patient_name[:7]}_0000.nii.gz'))
    #     shutil.copy(label_file, join(labelstr, train_patient_name))
    #     train_patient_names.append(train_patient_name)

    # test_patients = subfiles(test_folder, join=False, suffix=".nii.gz")
    # for p in test_patients:
    #     p = p[:-7]
    #     image_file = join(test_folder, p + ".nii.gz")
    #     serial_number = int(p[3:7])
    #     test_patient_name = f'{prefix}_{serial_number:03d}.nii.gz'
    #     shutil.copy(image_file, join(imagests, f'{test_patient_name[:7]}_0000.nii.gz'))
    #     test_patient_names.append(test_patient_name)

    print(train_patient_names, label_patient_names)
    json_dict = OrderedDict()
    json_dict['name'] = "FLARE2022"
    json_dict['description'] = "Fast and Low-resource semi-supervised Abdominal oRgan sEgmentation in CT (FLARE 2022)"
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "https://flare22.grand-challenge.org/Home/"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = OrderedDict({
        "00": "background",
        "01": "liver",
        "02": "right kidney",
        "03": "spleen",
        "04": "pancreas",
        "05": "aorta",
        "06": "inferior vena cava",
        "07": "right adrenal gland",
        "08": "left adrenal gland",
        "09": "gallbladder",
        "10": "esophagus",
        "11": "stomach",
        "12": "duodenum",
        "13": "left kidney"
        }
    )
    json_dict['numTraining'] = len(train_patient_names)
    # json_dict['numTest'] = len(test_patient_names)
    json_dict['training'] = [{"image": "./imagesTr/%s" % train_patient_name, "label": "./labelsTr/%s" % label_patient_name} 
                             for train_patient_name, label_patient_name 
                                in zip(train_patient_names, label_patient_names)]
    # json_dict['test'] = ["./imagesTs/%s" % test_patient_name for test_patient_name in test_patient_names]
    print(json_dict)
    save_json(json_dict, os.path.join(base, "dataset.json"))
