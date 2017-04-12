#!/bin/bash

# mkdir -p images/mnist/gen
echo "Making the Guassian Noise Exploration Directory"

mkdir -p images/mnist/gen/gaussAdv

for i in `seq 1 125`;
do
	echo th adversarial.lua -mnist -seed $i -mc -conv -cuda -out gaussAdv
	th adversarial.lua -mnist -seed $i -mc -conv -cuda -out gaussAdv
done 

mv ./images/mnist/results_from_adversarial.csv ./images/mnist/results_adv.csv 

echo "Making the Guassian Noise Exploration  of Original Image Directory"
mkdir -p -v images/mnist/gen/gaussOrig

for i in `seq 1 125`;
do
	echo th adversarial.lua -mnist -seed $i -mc -orig -conv -cuda -out gaussOrig
	th adversarial.lua -mnist -seed $i -mc -orig -conv -cuda -out gaussOrig
done 

mv ./images/mnist/results_from_original.csv ./images/mnist/results_orig.csv 

echo "Making the Nonparametric Noise Exploration Directory"
mkdir -p -v images/mnist/gen/histAdv

for i in `seq 1 125`;
do
	echo th adversarial.lua -mnist -seed $i -hist -mc -conv -cuda -out histAd
	th adversarial.lua -mnist -seed $i -hist -mc -conv -cuda -out histAd
done 

mv ./images/mnist/results_from_adversarial.csv ./images/mnist/results_adv_hist.csv 

echo "Making the Nonparametric Noise of Original Image Exploration Directory"
mkdir -p -v images/mnist/gen/histOrig

for i in `seq 1 125`;
do
	echo th adversarial.lua -mnist -seed $i -hist -mc -orig -conv -cuda -out histOrig
	th adversarial.lua -mnist -seed $i -hist -mc -orig -conv -cuda -out histOrig
done 

mv ./images/mnist/results_from_original.csv ./images/mnist/results_orig_hist.csv 
