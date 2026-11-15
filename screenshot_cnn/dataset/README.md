# dataset/ — not committed

The 280 heat-map screenshots (238 MB) that this experiment trains on are deliberately
kept out of git. Place them here as seven class folders:

```
dataset/
├── Normal/     *.png
├── Backward/   *.png
├── Forward/    *.png
├── Right/      *.png
├── Left/       *.png
├── Slouch/     *.png
└── Unknown/    *.png
```

Filenames must keep their original macOS screenshot timestamps
(`Screenshot 2026-11-15 at 3.08.55 PM.png`) — recording-session detection, and therefore
the entire train/val/test split, is derived from them.

You only need these to re-run `inspect_dataset.py` and `prepare_dataset.py`. The decoded
inputs are committed in `../processed_dataset/`, so `train.py` and `evaluate.py` work on
a fresh clone without them.
