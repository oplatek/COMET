#!/bin/bash
#
# expected to be run by gpu-bash run.sh on slurm
CMD="python"  # if forexample "echo python" ... then it is dry run, python if on GPU machine, gpu-python if on ufal cluster
MAIN_CFG="da/configs/cometkiwi_finetuning.yaml"
TRAINER_CFG="da/configs/trainer.yaml"
EARLY_STOP_CFG="da/configs/early_stopping.yaml"
CKPT_CFG="da/configs/model_checkpoint.yaml"
CKPT="$HOME/.cache/huggingface/hub/models--Unbabel--wmt22-cometkiwi-da/snapshots/b3a8aea5a5fc22db68a554b92b3d96eb6ea75cc9/checkpoints/model.ckpt"
USE_FIRST_LAYERS="null"   # options are null, -1 for all except last layers 24 effectively the same if there are 25 layers
SKIP_TRAINING="false"
EXPERIMENT_ID=""
RUN_NAME=""

debug=false  # valid optoins are false and true

. parse_options.sh

if [[ $debug == "true" ]] ; then
  set -x
fi

GIT_COMMIT="$(git rev-parse --short HEAD)"
TIMESTAMP="$(date -u +'%Y%m%dT%H%M%S')"

if [[ -z $RUN_NAME ]] ; then
  RUN_NAME="$(xkcd-exp)."
fi

if [[ -z "$EXPERIMENT_ID" ]] ; then
  EXPERIMENT_ID="Z_$TIMESTAMP.${RUN_NAME}$SLURM_JOB_ID.$GIT_COMMIT"
else
  printf "\nWARNING: you are setting experiment id to $EXPERIMENT_ID\nMake sure that not multiple jobs are overwriting each other!\n\n"
fi


if [[ $SKIP_TRAINING == true ]] ; then
  printf "WARNING training is not running expecting that you have setup $EXPERIMENT_ID so the exp/$EXPERIMENT_ID contains cfg.json and checkpoints\n\n"
  if [[ ! -d exp/$EXPERIMENT_ID/ ]] ; then printf "exp/$EXPERIMENT_ID/ does not exits! error 1" ; exit 1 ; fi
else
mkdir -p "exp/$EXPERIMENT_ID/"

sed \
  -e "s|use_first_layers: null|use_first_layers: ${USE_FIRST_LAYERS}|" \
  $MAIN_CFG \
  > "exp/$EXPERIMENT_ID/$(basename $MAIN_CFG)"
cat $TRAINER_CFG > "exp/$EXPERIMENT_ID/$(basename $TRAINER_CFG)"
cat $EARLY_STOP_CFG > "exp/$EXPERIMENT_ID/$(basename $EARLY_STOP_CFG)"
sed \
  -e "s|dirpath: TODO-dirpath|dirpath: exp/$EXPERIMENT_ID|" \
  $CKPT_CFG \
    > "exp/$EXPERIMENT_ID/$(basename $CKPT_CFG)"

printf "Applied changes to config 'exp/$EXPERIMENT_ID/$(basename $CKPT_CFG)'"

# hack https://github.com/Lightning-AI/pytorch-lightning/issues/5225#issuecomment-750032030
# how to avoid pytorch lightning messing with slurm -requining - AFAIK - PL still messes with SLURM
export UNUSED_SLURM_NTAKS=$SLURM_NTASK
export UNUSED_SLURM_JOB_NAME=$SLURM_JOB_NAME
unset SLURM_NTASK SLURM_JOB_NAME

if [[ -z $SLURM_JOB_ID ]] ; then printf "Not a SLURM job" ; exit 1; fi

$CMD \
  ./comet/cli/train.py \
  --run_name $RUN_NAME \
  --load_from_checkpoint "$CKPT" \
  --cfg "exp/$EXPERIMENT_ID/$(basename $MAIN_CFG)"
fi  # end of skip training: if [[ $SKIP_TRAINING == true ]] ; then ... fi

# TODO choose the best one

# If you want just one model TODO select just the best one
models="$(find exp/$EXPERIMENT_ID/ -name '*.ckpt' | head -n1)"

for model in $models ; do

  if [[ -z $SLURM_JOB_ID ]] ; then printf "Not a SLURM job" ; exit 1; fi
  $CMD \
    ./score_comet.py \
      -d ../efficient_pruning/vilem/data/jsonl/test.jsonl \
      -m $model \
      -o ${model}.out.json


  # for human score
  python3 eval_comet.py -d1 ${model}.out.json -d2 score \
    | tee ${model}.out.json.human.kendalltau
  # for cometkiwi correlation
  python3 eval_comet.py -d1 ${model}.out.json -d2 wmt22-cometkiwi-da.json \
    | tee ${model}.out.json.wmt2-cometkiwi-da.kendalltau

done # for model in $models 
