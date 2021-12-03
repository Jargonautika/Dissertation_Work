#!/usr/bin/env bash

# This script sets up and trains a series of regressors and classifiers 
# using the INTERSPEECH 2020 ADReSS Challenge data. 
# See https://arxiv.org/abs/2004.06833.

env=$1
scripts_dir=$2
data_dir=$3
dev_size=$4
random_state=$5
subset_folder=$6
extraction_dir=$7
extraction_script=$8
algorithm=$9
RUNNUM=${10}
exp_dir=${11:-""}
byFrame=${12:-"True"}
stage=${13:-1}
crossVal=${14:-"True"} # True = LOSO; False = WITHHELD

source $env
if [ ! -d $exp_dir ]; then
	mkdir $exp_dir
fi
if [ "$exp_dir" = "" ]; then
	exp_dir=$(mktemp -d)
fi

# Split training data into training and development sets
# rm -fr $exp_dir/*
# exit 0
mkdir -p $exp_dir/reports/plots/histograms $exp_dir/reports/plots/vowelSpace
mkdir -p $exp_dir/vectors/classifiers $exp_dir/vectors/regressors $exp_dir/vectors/pca
mkdir -p $exp_dir/models/classifiers $exp_dir/models/regressors
mkdir -p $exp_dir/data/train/csv/30 $exp_dir/data/dev/csv/30
mkdir -p $exp_dir/data/train/csv/50 $exp_dir/data/dev/csv/50
mkdir -p $exp_dir/data/train/csv/70 $exp_dir/data/dev/csv/70
mkdir -p $exp_dir/data/train/wav $exp_dir/data/dev/wav
mkdir -p $exp_dir/reports/plots/confusionMatrices
mkdir -p $exp_dir/reports/plots/linearRegressionPlots
mkdir -p $exp_dir/data/acoustics

echo "$algorithm-$RUNNUM" > $exp_dir/algorithm.txt

# Run Feature Extraction
if [[ $stage -le 0 ]]; then

	if [ "$algorithm" = "auditory-local" ]; then
		cp /home/chasea2/SPEECH/Dissertation_Work/Experiments/Code/PYTHON/AUDITORY/extractor_local.py /home/chasea2/SPEECH/Dissertation_Work/Experiments/Code/PYTHON/AUDITORY/extractor.py
	fi

	echo "Now working on extracting $algorithm"

	# TODO fix this for LOSO
	if [ "$crossVal" = "True" ]; then

		# Beginning of implementation for cross-validation using the training and dev sets only
		for condition in cc cd; do
			metadata_loc=$data_dir/${condition}_meta_data.txt
			python3 $scripts_dir/train-dev-split $metadata_loc $exp_dir $condition $dev_size $random_state 
		done

	fi

	if [ "$crossVal" = "False" ]; then

		# Beginning implementation for withheld test set
		for condition in cc cd; do
			metadata_loc=$data_dir/${condition}_meta_data.txt
			python3 $scripts_dir/train-dev-split-2 $metadata_loc $exp_dir $condition "train"
		done

		meta_data_test=/home/chasea2/SPEECH/Dissertation_Work/Experiments/Data/ADReSS-IS2020-data/meta_data_test.txt
		python3 $scripts_dir/train-dev-split-2 $meta_data_test $exp_dir $condition 'test'
	fi

	# Combine the train and dev frames
	for dir in train dev; do
		python3 $scripts_dir/combine-dataframes $exp_dir/data/$dir
	done

	# Collect the data together and extract actually good waveforms
	for dir in train dev; do
		python3 $scripts_dir/collect-data-2 $exp_dir $data_dir $subset_folder $dir $crossVal
	done

	# Normalize the data
	python3 $scripts_dir/normalize_rms $exp_dir
	# exit 0

	# Extract features to figure out how big to make the matrix
	if [ "$algorithm" = "auditory-global" ] || [ "$algorithm" = "auditory-local" ]; then
		for dir in train dev; do 
			python3 $scripts_dir/$extraction_script $exp_dir $extraction_dir $exp_dir/data/$dir $algorithm
		done
	else
		for dir in train dev; do 
			python3 $scripts_dir/$extraction_script $exp_dir $extraction_dir $exp_dir/data/$dir
		done
	fi

	echo "Done extracting features for $algorithm"
	if [ "$algorithm" = "auditory-global" ] || [ "$algorithm" = "auditory-local" ]; then
		rm /home/chasea2/SPEECH/Dissertation_Work/Experiments/Code/PYTHON/AUDITORY/extractor.py
	fi

	exit 0
	
fi

# Train and Evaluate the models
if [[ $stage -le 1 ]]; then

	# Regression
	# echo "Now working on regressing $algorithm"
	# python3 $scripts_dir/regression/main.py $exp_dir $data_dir $random_state $byFrame $RUNNUM $crossVal $algorithm

	# Classification
	echo "Now working on classifying $algorithm"
	python3 $scripts_dir/classification/main.py $exp_dir $data_dir $random_state $byFrame $RUNNUM $crossVal $algorithm

	echo "Done doing basic machine learning with $algorithm for training set"
	exit 0

fi

# Train and Evaluate Neural Network
if [[ $stage -le 2 ]]; then

	# Feed Forward Neural Network
	env=/home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Envs/NEURAL/bin/activate
	source $env
	mkdir -p $exp_dir/nn_checkpoints

	# echo "Now working on the feed-forward neural net for $algorithm"
	# python3 $scripts_dir/neural/Keras/main.py $exp_dir $data_dir $algorithm $random_state $byFrame $RUNNUM $crossVal

	echo "Now working on the GRU Neural Net for $algorithm"
	# python3 $scripts_dir/neural/PyTorch/main.py $exp_dir $data_dir $random_state $byFrame $RUNNUM $crossVal # Old non-working feed forward neural net; replaced with Keras early 2021
	python3 $scripts_dir/neural/PyTorch/GRU.py $exp_dir $data_dir $random_state $byFrame $RUNNUM $crossVal

	echo "Done doing neural network modeling with $algorithm"
	exit 0

fi

deactivate
