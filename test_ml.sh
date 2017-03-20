PRE=grid
OPTIMIZER=ls
MODEL=ml

epoch=5
batch_size=128
lr=0.8


X=$1
LOAD=$2
MEAN=$3
SIGMA=$4
Y=$5

python main.py --task test --X $X --model $MODEL --pre $PRE --epoch $epoch --batch_size $batch_size --lr $lr --output $Y --load $LOAD --mean $MEAN --sigma $SIGMA

