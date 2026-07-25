@echo off
REM Train official Isaac Lab Franka Reach (NVIDIA GPU + Isaac Lab required).
REM Run from the IsaacLab repository root on a Windows GPU machine.
REM See docs\cloud_setup.md

set TASK=Isaac-Reach-Franka-v0
if not "%~1"=="" set TASK=%~1

if not exist "isaaclab.bat" (
  echo [error] Run this from the IsaacLab repository root ^(isaaclab.bat not found^).
  exit /b 1
)

echo [info] task=%TASK%
call isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task %TASK% --headless --seed 42
echo [ok] training finished. Check logs\rsl_rl\%TASK%\ for checkpoints.
