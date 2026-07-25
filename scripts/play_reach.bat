@echo off
REM Play Franka Reach in Isaac Lab (GPU required). Run from IsaacLab root.

set TASK=Isaac-Reach-Franka-Play-v0
if not "%~1"=="" set TASK=%~1

if not exist "isaaclab.bat" (
  echo [error] Run this from the IsaacLab repository root.
  exit /b 1
)

call isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task %TASK% --num_envs 16 --use_pretrained_checkpoint
echo [ok] play finished.
