#!/bin/bash
#
# expected to be run by gpu-bash run.sh on slurm
CMD="python"  # if forexample "echo python" ... then it is dry run, python if on GPU machine, gpu-python if on ufal cluster
MAIN_CFG="da/configs/cometkiwi_finetuning.yaml"
TRAINER_CFG="da/configs/trainer.yaml"
EARLY_STOP_CFG="da/configs/early_stopping.yaml"
CKPT_CFG="da/configs/model_checkpoint.yaml"
CKPT="$HOME/.cache/huggingface/hub/models--Unbabel--wmt22-cometkiwi-da/snapshots/b3a8aea5a5fc22db68a554b92b3d96eb6ea75cc9/checkpoints/model.ckpt"

debug=false

. parse_options.sh

if [[ $debug == "true" ]] ; then
  set -x
fi

GIT_COMMIT="$(git rev-parse --short HEAD)"
TIMESTAMP="$(date -u +'%Y%m%dT%H%M%S')"

if [[ -z $SLURM_JOB_ID ]] ; then
  printf "Not a SLURM job" ; exit 1
fi
EXPERIMENT_ID="Z_$TIMESTAMP.$SLURM_JOB_ID.$GIT_COMMIT"


mkdir -p "exp/$EXPERIMENT_ID/"

cat $MAIN_CFG > "exp/$EXPERIMENT_ID/$(basename $MAIN_CFG)"
cat $TRAINER_CFG > "exp/$EXPERIMENT_ID/$(basename $TRAINER_CFG)"
cat $EARLY_STOP_CFG > "exp/$EXPERIMENT_ID/$(basename $EARLY_STOP_CFG)"
sed \
  -e "s;dirpath: TODO-dirpath;dirpath: exp/$EXPERIMENT_ID;" \
  $CKPT_CFG \
    > exp/$EXPERIMENT_ID/$(basename $CKPT_CFG)

# hack https://github.com/Lightning-AI/pytorch-lightning/issues/5225#issuecomment-750032030
# how to avoid pytorch lightning messing with slurm -requining - AFAIK - PL still messes with SLURM
export UNUSED_SLURM_NTAKS=$SLURM_NTASK
export UNUSED_SLURM_JOB_NAME=$SLURM_JOB_NAME
unset SLURM_NTASK SLURM_JOB_NAME

$CMD \
  ./comet/cli/train.py \
  --load_from_checkpoint "$CKPT" \
  --cfg "exp/$EXPERIMENT_ID/$(basename $MAIN_CFG)"