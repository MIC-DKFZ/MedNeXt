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

import numpy as np
from nnunet_mednext.experiment_planning.experiment_planner_baseline_3DUNet_v21 import ExperimentPlanner3D_v21
from nnunet_mednext.experiment_planning.experiment_planner_baseline_2DUNet_v21 import ExperimentPlanner2D_v21
from nnunet_mednext.paths import *


class ExperimentPlanner3D_v21_customTargetSpacing_2x2x2(ExperimentPlanner3D_v21):
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlanner3D_v21, self).__init__(folder_with_cropped_data, preprocessed_output_folder)
        # we change the data identifier and plans_fname. This will make this experiment planner save the preprocessed
        # data in a different folder so that they can co-exist with the default (ExperimentPlanner3D_v21). We also
        # create a custom plans file that will be linked to this data
        self.data_identifier = "nnUNetData_plans_v2.1_trgSp_2x2x2"
        self.plans_fname = join(self.preprocessed_output_folder,
                                "nnUNetPlansv2.1_trgSp_2x2x2_plans_3D.pkl")

    def get_target_spacing(self):
        # simply return the desired spacing as np.array
        return np.array([2., 2., 2.]) # make sure this is float!!!! Not int!


class ExperimentPlanner3D_v21_customTargetSpacing_1x1x1(ExperimentPlanner3D_v21):
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlanner3D_v21, self).__init__(folder_with_cropped_data, preprocessed_output_folder)
        # we change the data identifier and plans_fname. This will make this experiment planner save the preprocessed
        # data in a different folder so that they can co-exist with the default (ExperimentPlanner3D_v21). We also
        # create a custom plans file that will be linked to this data
        self.data_identifier = "nnUNetData_plans_v2.1_trgSp_1x1x1"
        self.plans_fname = join(self.preprocessed_output_folder,
                                "nnUNetPlansv2.1_trgSp_1x1x1_plans_3D.pkl")

    def get_target_spacing(self):
        # simply return the desired spacing as np.array
        return np.array([1., 1., 1.]) # make sure this is float!!!! Not int!
    
    def get_properties_for_stage(
        self, current_spacing, original_spacing, original_shape, num_cases, num_modalities, num_classes
    ):
        """
        ExperimentPlanner configures pooling so that we pool late. Meaning that if the number of pooling per axis is
        (2, 3, 3), then the first pooling operation will always pool axes 1 and 2 and not 0, irrespective of spacing.
        This can cause a larger memory footprint, so it can be beneficial to revise this.

        Here we are pooling based on the spacing of the data.

        """
        plans = super(ExperimentPlanner3D_v21_customTargetSpacing_1x1x1, self).get_properties_for_stage(
            current_spacing, original_spacing, original_shape, num_cases, num_modalities, num_classes
        )
        plans["patch_size"] = [128, 128, 128]
        return plans


class ExperimentPlanner2D_v21_customTargetSpacing_1x1x1(ExperimentPlanner2D_v21):
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlanner2D_v21, self).__init__(folder_with_cropped_data, preprocessed_output_folder)
        # we change the data identifier and plans_fname. This will make this experiment planner save the preprocessed
        # data in a different folder so that they can co-exist with the default (ExperimentPlanner3D_v21). We also
        # create a custom plans file that will be linked to this data
        self.data_identifier = "nnUNetData2D_plans_v2.1_trgSp_1x1x1"
        self.plans_fname = join(self.preprocessed_output_folder,
                                "nnUNetPlansv2.1_trgSp_1x1x1_plans_2D.pkl")

    def get_target_spacing(self):
        # simply return the desired spacing as np.array
        return np.array([1., 1., 1.]) # make sure this is float!!!! Not int!
    
    def get_properties_for_stage(
        self, current_spacing, original_spacing, original_shape, num_cases, num_modalities, num_classes
    ):
        plans = super(ExperimentPlanner2D_v21_customTargetSpacing_1x1x1, self).get_properties_for_stage(
            current_spacing, original_spacing, original_shape, num_cases, num_modalities, num_classes
        )
        plans["patch_size"] = [512, 512]
        return plans