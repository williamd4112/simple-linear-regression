#!/bin/sh

PRE=grid
X=data/X_train.csv
Y=data/T_train.csv
K=2
OPTIMIZER=seq
SCALE=1.0
FRAC=0.8
MODEL=map

for epoch in $1
do
    for batch_size in $2
    do    
        for lr in $3
        do
            # kmeans: number of cluster; grid: gsize
            for d in $4
            do
                for alpha in $5
                do
                    log_name="log/${MODEL}-m0-${m0}-s0-${s0}-beta-${beta}-${PRE}-${d}.log"
                    python main.py --task train --X $X --Y $Y  --K $K --model ml --pre $PRE --d $d --gsize $d --optimizer $OPTIMIZER --scale $SCALE --frac $FRAC --epoch $epoch --batch_size $batch_size --lr $lr --log $log_name
                done
            done
        done
    done
done
