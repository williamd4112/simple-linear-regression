PRE=grid
X=data/X_train.csv
Y=data/T_train.csv
K=1
OPTIMIZER=seq
SCALE=1.0
MODEL=map
D=0.1

epoch=10
batch_size=128
lr=0.5
alpha=0.1

OUTPUT="model/${MODEL}-epoch-${epoch}-${batch_size}-lr-${lr}-${PRE}-${D}"

FRAC=$1

python main.py --task train --X $X --Y $Y  --K $K --model $MODEL --pre $PRE --gsize $D --d $D --optimizer $OPTIMIZER --scale $SCALE --frac $FRAC --epoch $epoch --batch_size $batch_size --lr $lr --alpha $alpha --output $OUTPUT --plot 2d

