#!/bin/bash

# List of database names
database_names=('Haptics' 'Worms' 'Computers' 'UWaveGestureLibraryAll' 'Strawberry' 'Car' 'BeetleFly' 'wafer' 'CBF' 'Adiac' 'Lighting2' 'ItalyPowerDemand' 'yoga' 'Trace' 'ShapesAll' 'Beef' 'MALLAT' 'MiddlePhalanxTW' 'Meat' 'Herring'
'MiddlePhalanxOutlineCorrect' 'FordA' 'SwedishLeaf' 'SonyAIBORobotSurface' 'InlineSkate' 'WormsTwoClass' 'OSULeaf' 'Ham' 'uWaveGestureLibrary_Z' 'NonInvasiveFatalECG_Thorax1' 'ToeSegmentation1' 'ScreenType' 'SmallKitchenAppliances' 'WordsSynonyms' 'MoteStrain' 'synthetic_control' 'Cricket_X' 'ECGFiveDays' 'Wine' 'Cricket_Y' 'TwoLeadECG' 'Two_Patterns' 'Phoneme' 'MiddlePhalanxOutlineAgeGroup' 'DistalPhalanxOutlineCorrect' 'DistalPhalanxTW' 'FacesUCR' 'ECG5000' '50words' 'HandOutlines' 'Coffee' 'Gun_Point' 'FordB' 'InsectWingbeatSound' 'MedicalImages' 'Symbols' 'ArrowHead' 'ProximalPhalanxOutlineAgeGroup' 'SonyAIBORobotSurfaceII' 'ChlorineConcentration' 'Plane' 'Lighting7' 'PhalangesOutlinesCorrect' 'ShapeletSim' 'DistalPhalanxOutlineAgeGroup' 'uWaveGestureLibrary_X' 'FaceFour' 'RefrigerationDevices' 'ECG200' 'ToeSegmentation2' 'CinC_ECG_torso' 'BirdChicken' 'OliveOil' 'LargeKitchenAppliances' 'uWaveGestureLibrary_Y' 'NonInvasiveFatalECG_Thorax2' 'FISH' 'ProximalPhalanxOutlineCorrect' 'Cricket_Z' 'FaceAll' 'StarLightCurves' 'ElectricDevices' 'Earthquakes' 'DiatomSizeReduction' 'ProximalPhalanxTW')

#selected for sota
database_names=('Computers' 'UWaveGestureLibraryAll' 'Strawberry' 'BeetleFly' 'wafer' 'CBF' 'Adiac' 'Lighting2' 'ItalyPowerDemand' 'yoga' 'Trace' 'ShapesAll' 'Beef' 'MALLAT' 'Herring'
'MiddlePhalanxOutlineCorrect' 'FordA' 'SwedishLeaf' 'FaceAll' 'Phoneme' 'StarLightCurves' 'ECG200' 'ECGFiveDays' 'OliveOil' 'MoteStrain' 'SonyAIBORobotSurface' 'SonyAIBORobotSurfaceII' 'Ham' 'NonInvasiveFatalECG_Thorax1' 'NonInvasiveFatalECG_Thorax2')


# Iterate over each database name and run the Python script
for database_name in "${database_names[@]}"
do
  python teacher_main.py --dataset=$database_name --seed_array 1001 --save_dir="teacher_models/" --common_dir='teacher_results/' --result_csv='results_teacher005.csv' --avg_result_csv='avg_results_teacher005.csv' --max_result_csv='max_results_teacher005.csv' --mid_channels=64 --epochs=500 --lr=0.005 --batch=32 --patience=50 
done
