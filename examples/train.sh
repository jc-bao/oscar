# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for OSCAR. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

# Script to automatically run training

####################################################################
###################### MACROS TO SET HERE ##########################

CONDA_PATH=~/miniconda3/envs/rlgpu/lib/:~/anaconda3/envs/rlgpu/lib/  # Should be absolute path to your conda oscar env lib directory
TASK=push                              # Options are: {trace, pour, push}
CONTROLLER=oscar                        # Options are: {oscar, osc, osc_no_vices, ik, joint_tor, joint_vel, joint_pos}
PRETRAINED_OSCAR=default                # If using OSCAR, should set this to absolute fpath to pretrained OSCAR .pth file. Setting this to "default" results in default pretrained model being loaded
EPOCHS=default                          # Manually set value (int), or "default" results in default value being applied

####################################################################
##### YOU SHOULDN'T HAVE TO TOUCH ANYTHING BELOW THIS POINT :) #####

# Setup env variables
export LD_LIBRARY_PATH=${CONDA_PATH}
export CUDA_VISIBLE_DEVICES=0

# Setup python interpreter
source activate rlgpu

# Get current working directory
temp=$( realpath "$0"  )
DIR=$(dirname "$temp")

# Setup local vars
OSCAR_PATH=${DIR}/..
CFG_PATH=${OSCAR_PATH}/oscar/cfg/train

# Run training

# Process macros
if [ ${CONTROLLER} == "oscar" ]
then
  if [ ${PRETRAINED_OSCAR} == "default" ]
  then
    PRETRAINED_OSCAR="${OSCAR_PATH}/trained_models/pretrain/Pretrace_oscar__seed_1.pth"
  fi
else
  PRETRAINED_OSCAR="None"
fi

if [ ${EPOCHS} == "default" ]
then
  EPOCHS=0
fi

# Determine device to use -- Pour must use CPU
DEVICE=""
if [ ${TASK} == "pour" ]
then
  DEVICE+="CPU"
else
  DEVICE+="GPU"
fi

# Potentially add header string
HEADER_STR=""
if [ ${CONTROLLER} == "osc_no_vices" ]
then
  HEADER_STR+="no_vices_"
fi

# Define config(s) for controller
CONTROLLER_CONFIGS="${CFG_PATH}/controller/${CONTROLLER}.yaml"
if [ ${CONTROLLER} == "oscar" ]
then
  CONTROLLER_CONFIGS+=" ${CFG_PATH}/controller/oscar_settings/delan_residual.yaml"
fi

# Determine agent to use for task
declare -A TASK_AGENT_MAPPING
TASK_AGENT_MAPPING[trace]="franka_pitcher"
TASK_AGENT_MAPPING[pour]="franka_pitcher"
TASK_AGENT_MAPPING[push]="franka"

# We procedurally load the cfgs necessary, in the order as follows:
# 1. Base cfg
# 2. Sim cfg
# 3. Agent cfg
# 4. Task cfg
# 5. Controller cfg
# 6. Additional cfg (e.g.: OSCAR-specific settings)

python ${OSCAR_PATH}/oscar/train.py \
--cfg_env ${CFG_PATH}/base.yaml \
--cfg_env_add \
${CFG_PATH}/sim/physx.yaml \
${CFG_PATH}/agent/${TASK_AGENT_MAPPING[${TASK}]}.yaml \
${CFG_PATH}/task/${TASK}.yaml \
${CONTROLLER_CONFIGS} \
--pretrained_delan ${PRETRAINED_OSCAR} \
--device ${DEVICE} \
--ppo_device GPU \
--headless \
--max_iterations ${EPOCHS} \
--logdir ${OSCAR_PATH}/log/train \
--experiment_name ${HEADER_STR}train \
--wandb
--save_video