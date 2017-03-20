#!/bin/sh

PRE=grid
X=data/X_train.csv
Y=data/T_train.csv
K=2
OPTIMIZER=ls
SCALE=1.0
FRAC=0.8
MODEL=bayes

for m0 in $1
do
    for s0 in $2
    do    
        for beta in $3
        do
            # kmeans: number of cluster; grid: gsize
            for d in $4
            do
                log_name="log/${MODEL}-m0-${m0}-s0-${s0}-beta-${beta}-${PRE}-${d}.log"
                python main.py --task train --X $X --Y $Y  --K $K --model bayes --pre $PRE --d $d --gsize $d --optimizer $OPTIMIZER --scale $SCALE --frac $FRAC --m0 $m0 --s0 $s0 --beta $beta --log $log_name
            done
        done
    done
done
