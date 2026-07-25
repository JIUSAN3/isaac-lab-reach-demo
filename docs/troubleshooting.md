# Troubleshooting (stuff I actually hit)

## `No module named 'isaaclab'`

Core package didn’t install. Often `flatdict` build failure with new setuptools.

```bash
pip install 'setuptools<70' wheel
pip install flatdict==4.0.1 --no-build-isolation
pip install -e source/isaaclab --no-build-isolation
pip install -e source/isaaclab_assets
pip install -e source/isaaclab_tasks
pip install -e "source/isaaclab_rl[rsl_rl]"
```

## Lab / Sim version mismatch

Default clone wanted Isaac Sim 6 + Python 3.12.  
For Sim 5.1 use Lab tag **`v2.3.2`**.

## `libXt.so.6` / `libGLU.so.1`

```bash
apt-get install -y libxt6 libglu1-mesa libx11-6 libxext6 libxrender1
```

## Disk full / slow extract

`isaacsim[extscache]` is huge. Use a big data disk; don’t install on a 30GB system overlay.  
On AutoDL, `/root/autodl-fs` was slow but had space.

## CUDA OOM

Lower `--num_envs` (64 → 32 → 16). Keep `--headless`.

## EULA prompt hangs headless

```bash
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y
# or pipe:  yes | ./isaaclab.sh -p ...
```

## Vulkan / no display warnings

Common on cloud headless. Train can still proceed for this reach setup.

## `pkg_resources` missing

```bash
pip install 'setuptools<81'
```
