# results

## Laptop mock

| File | |
|------|--|
| `mock_reach_metrics.*` | scripted controller |
| `mock_reach_random.*` | random baseline |

```bash
python scripts/run_mock_reach.py --episodes 5 --seed 42
```

## Cloud Isaac

| File | |
|------|--|
| `isaac_run_notes.md` | what I actually ran / fixed |
| `isaac_reach_train_curve.png` | mean reward curve |
| `isaac_cloud/` | checkpoint + env/agent yaml + scalars |

Raw multi-GB Isaac installs stay on the cloud disk, not in git.
