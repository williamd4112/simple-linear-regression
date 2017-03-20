PRE=grid
MODEL=ml

LOAD=$1
MEAN=$2
SIGMA=$3
PLOT_TYPE=$4

python main.py --task plot  --model $MODEL --pre $PRE --load $LOAD --mean $MEAN --sigma $SIGMA --plot $PLOT_TYPE

