#!/bin/bash

# List of database names
database_names_all=('Haptics' 'Worms' 'Computers' 'UWaveGestureLibraryAll' 'Strawberry' 'Car' 'BeetleFly' 'wafer' 'CBF' 'Adiac' 'Lighting2' 'ItalyPowerDemand' 'yoga' 'Trace' 'ShapesAll' 'Beef' 'MALLAT' 'MiddlePhalanxTW' 'Meat' 'Herring'
'MiddlePhalanxOutlineCorrect' 'FordA' 'SwedishLeaf' 'SonyAIBORobotSurface' 'InlineSkate' 'WormsTwoClass' 'OSULeaf' 'Ham' 'uWaveGestureLibrary_Z' 'NonInvasiveFatalECG_Thorax1' 'ToeSegmentation1' 'ScreenType' 'SmallKitchenAppliances' 'WordsSynonyms' 'MoteStrain' 'synthetic_control' 'Cricket_X' 'ECGFiveDays' 'Wine' 'Cricket_Y' 'TwoLeadECG' 'Two_Patterns' 'Phoneme' 'MiddlePhalanxOutlineAgeGroup' 'DistalPhalanxOutlineCorrect' 'DistalPhalanxTW' 'FacesUCR' 'ECG5000' '50words' 'HandOutlines' 'Coffee' 'Gun_Point' 'FordB' 'InsectWingbeatSound' 'MedicalImages' 'Symbols' 'ArrowHead' 'ProximalPhalanxOutlineAgeGroup' 'SonyAIBORobotSurfaceII' 'ChlorineConcentration' 'Plane' 'Lighting7' 'PhalangesOutlinesCorrect' 'ShapeletSim' 'DistalPhalanxOutlineAgeGroup' 'uWaveGestureLibrary_X' 'FaceFour' 'RefrigerationDevices' 'ECG200' 'ToeSegmentation2' 'CinC_ECG_torso' 'BirdChicken' 'OliveOil' 'LargeKitchenAppliances' 'uWaveGestureLibrary_Y' 'NonInvasiveFatalECG_Thorax2' 'FISH' 'ProximalPhalanxOutlineCorrect' 'Cricket_Z' 'FaceAll' 'StarLightCurves' 'ElectricDevices' 'Earthquakes' 'DiatomSizeReduction' 'ProximalPhalanxTW')
'ShapesAll' 'Beef' 'MALLAT' 'Herring'
'MiddlePhalanxOutlineCorrect' 'FordA' 'SwedishLeaf')

#selected for sota
database_names=('Computers' 'UWaveGestureLibraryAll' 'Strawberry' 'BeetleFly' 'wafer' 'CBF' 'Adiac' 'Lighting2' 'ItalyPowerDemand' 'yoga' 'Trace' 'ShapesAll' 'Beef' 'MALLAT' 'Herring'
'MiddlePhalanxOutlineCorrect' 'FordA' 'SwedishLeaf' 'FaceAll' 'Phoneme' 'StarLightCurves' 'ECG200' 'ECGFiveDays' 'OliveOil' 'MoteStrain' 'SonyAIBORobotSurface' 'SonyAIBORobotSurfaceII' 'Ham' 'NonInvasiveFatalECG_Thorax1' 'NonInvasiveFatalECG_Thorax2')


# Iterate over each database name and run the Python script
for database_name in "${database_names[@]}"
do 
    python student_main.py --dataset=$database_name --seed_array 1 24 300 49000 1001 --ratios 0.1 0.5 1 10 100 --save_dir="student_models/" --patience=50 --teacher_load="teacher_models_bulk/" --memory_loss_ratio=0 --hinton_loss_ratio=0 --task_ratio=1  --bench_rkd_dist_ratio=0 --bench_rkd_angle_ratio=0 --bench_fitnet_ratio=0  --bench_attention_ratio=0 --bench_temp_dist_ratio=0 --bench_SaliencyKD_ratio=0 --result_csv='SalKD_results_student2_temp8.csv' --avg_result_csv='SalKD_avg_results_student2_temp8.csv' --max_result_csv='SalKD_max_results_student2_temp8.csv' --common_dir='student_res/' --mid_channels_student=2 --epochs=500 --lr=0.005 --batch=32 --temperature=8 --num_samples=50 --sub_seq_len=10 

done


