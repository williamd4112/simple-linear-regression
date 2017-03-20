PRE=grid
OPTIMIZER=ls
MODEL=ml

X=$1
LOAD=$2
MEAN=$3
SIGMA=$4
Y=$5

python main.py --task test --X $X --model $MODEL --pre $PRE --output $Y --load $LOAD --mean $MEAN --sigma $SIGMA

