

for fold in 0 1 2 3 4
do
    bsub -R "tensorcore" -R "select[hname!='e230-dgxa100-1']" -R "select[hname!='e230-dgxa100-2']" -R "select[hname!='e230-dgxa100-3']" -R "select[hname!='e230-dgxa100-4']" -R "select[hname!='e230-dgx2-2']" -R "select[hname!='e230-dgx2-1']" -R "select[hname!='e230-dgx1-1']" -L /bin/bash -g /s539y/cal_metrics -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=38G -q gpu -J mednext_d1_k3_grn_hippo_$fold -u "saikat.roy@dkfz-heidelberg.de" -B -N "source ~/.bashrc && conda activate pytorch20_env && python ~/PythonProjects/MedNeXt/nnunet_mednext/run/run_training.py 3d_fullres nnUNetTrainerV2_MedNeXt_GRN_L_kernel3 Task004_Hippocampus $fold -p nnUNetPlansv2.1_trgSp_1x1x1"
done