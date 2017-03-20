PRE=grid
K=1
OPTIMIZER=ls
SCALE=1.0
FRAC=0.8
MODEL=bayes
D=0.15

m0=0.0
s0=2.0
beta=25.0

#OUTPUT="${MODEL}-m0-${m0}-s0-${s0}-beta-${beta}-${PRE}-${D}.csv"


X=$1
LOAD=$2
MEAN=$3
SIGMA=$4
Y=$5

python main.py --task test --X $X --model $MODEL --pre $PRE --m0 $m0 --s0 $s0 --beta $beta --output $Y --load $LOAD --mean $MEAN --sigma $SIGMA

