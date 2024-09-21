#!/bin/bash
MAIN_CFG="da/configs/cometkiwi_finetuning.yaml"
CKPT="$HOME/.cache/huggingface/hub/models--Unbabel--wmt22-cometkiwi-da/snapshots/b3a8aea5a5fc22db68a554b92b3d96eb6ea75cc9/checkpoints/model.ckpt"
sed \
   -e 's/trainer: trainer.yaml/trainer: debugging_trainer.yaml/' \
    $MAIN_CFG > $MAIN_CFG.debug.yaml
python ./comet/cli/train.py --load_from_checkpoint "$CKPT" --cfg $MAIN_CFG.debug.yaml
