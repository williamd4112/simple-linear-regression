PRE=grid
X=data/X_train.csv
Y=data/T_train.csv
K=1
OPTIMIZER=ls
SCALE=0.75
MODEL=bayes
D=0.015

m0=0.0
s0=2.0
beta=100.0

OUTPUT="model/${MODEL}-m0-${m0}-s0-${s0}-beta-${beta}-${PRE}-${D}"

FRAC=$1

python main.py --task train --X $X --Y $Y  --K $K --model $MODEL --pre $PRE --gsize $D --d $D --optimizer $OPTIMIZER --scale $SCALE --frac $FRAC --m0 $m0 --s0 $s0 --beta $beta --output $OUTPUT --plot 2d

