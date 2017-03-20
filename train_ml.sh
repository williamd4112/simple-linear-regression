PRE=grid
X=data/X_train.csv
Y=data/T_train.csv
K=1
OPTIMIZER=seq
SCALE=1.0
MODEL=ml
D=0.015

epoch=10
batch_size=128
lr=0.5

OUTPUT="model/${MODEL}-epoch-${epoch}-${batch_size}-lr-${lr}-${PRE}-${D}"

FRAC=$1

python main.py --task train --X $X --Y $Y  --K $K --model $MODEL --pre $PRE --gsize $D --d $D --optimizer $OPTIMIZER --scale $SCALE --frac $FRAC --epoch $epoch --batch_size $batch_size --lr $lr --output $OUTPUT --plot 2d

