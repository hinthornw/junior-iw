#!/bin/bash
for i in `seq 1 250`;
do
	th adversarial.lua -cuda -mnist -seed $i 
done 

mv ./images/mnist/results_from_adversarial.csv ./images/mnist/results_adv.csv 

# for i in `seq 1 125`;
# do
# 	th adversarial.lua -cuda -mnist -seed $i -mc -orig -out gaussOrig
# done 

# mv ./images/mnist/results_from_original.csv ./images/mnist/results_orig.csv 

# for i in `seq 1 125`;
# do
# 	th adversarial.lua -cuda -mnist -seed $i -hist -mc -out histAdv
# done 

# mv ./images/mnist/results_from_adversarial.csv ./images/mnist/results_adv_hist.csv 

# for i in `seq 1 125`;
# do
# 	th adversarial.lua -cuda -mnist -seed $i -hist -mc -orig - out histOrig
# done 

# mv ./images/mnist/results_from_original.csv ./images/mnist/results_orig_hist.csv 
