#!/usr/bin/env bash

# The purpose of this script is to combine the features extracted from
# the various algorithms tried individually (AMS, MFCC, RASTA) and run
# a series of regressors and classifiers on the features combined together

RUNNUM=${1:-3104}
exp_dir=${2:-/tmp/tmp.Wy3D2NOZ86}
nextStage=${3:-0}
crossVal=${4:-"True"}
stage=${5:-1}

for dir in /tmp/tmp.*; do
	if [ -f $dir/algorithm.txt ]; then
		if grep -q "ams-$RUNNUM" $dir/algorithm.txt; then ams_exp_dir=$dir
		elif grep -q "mfcc-$RUNNUM" $dir/algorithm.txt; then mfcc_exp_dir=$dir
		elif grep -q "rasta-$RUNNUM" $dir/algorithm.txt; then rasta_exp_dir=$dir
		fi
	fi
done

scripts_dir=../scripts
data_dir=../../Data/ADReSS-IS2020-data/train
random_state=1

source /home/chasea2/SPEECH/Adams_Chase_Preliminary_Exam/Experiments/Envs/COMBINE/bin/activate
if [ "$exp_dir" = "" ]; then
	exp_dir=$(mktemp -d)
fi

mkdir -p $exp_dir/vectors/classifiers $exp_dir/vectors/regressors
mkdir -p $exp_dir/models/classifiers $exp_dir/models/regressors
mkdir -p $exp_dir/data/train/wav $exp_dir/data/dev/wav
mkdir -p $exp_dir/data/train/csv $exp_dir/data/dev/csv
mkdir -p $exp_dir/reports/plots
mkdir -p $exp_dir/reports
echo "combine-$RUNNUM" > $exp_dir/algorithm.txt

if [[ $stage -le 0 ]]; then

	# Collect the data together
	for ddir in train dev; do
		python3 $scripts_dir/concatenate-features $exp_dir $ams_exp_dir $mfcc_exp_dir $rasta_exp_dir $ddir
	done
	# exit 0

fi

if [[ $stage -le 1 ]]; then

	# Run the ML or NN algorithm across the combined data
	bash ./configs/combine $RUNNUM $exp_dir $nextStage $crossVal # We can't use $stage here because they're not aligned from combine.sh to train.sh 

fi

deactivate
