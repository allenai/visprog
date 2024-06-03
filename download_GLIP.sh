#!/bin/bash

# change this to your preferred download location
PRETRAINED_MODELS_PATH=./pretrained_models

# GLIP model
mkdir -p $PRETRAINED_MODELS_PATH/GLIP/checkpoints
mkdir -p $PRETRAINED_MODELS_PATH/GLIP/configs
wget -nc -P $PRETRAINED_MODELS_PATH/GLIP/checkpoints https://huggingface.co/GLIPModel/GLIP/resolve/main/glip_large_model.pth
wget -nc -P $PRETRAINED_MODELS_PATH/GLIP/configs https://raw.githubusercontent.com/microsoft/GLIP/main/configs/pretrain/glip_Swin_L.yaml
