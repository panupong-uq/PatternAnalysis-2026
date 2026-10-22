"""Predict with the trained HipMRI models: figures, failure cases and inference time.

Run this after ``train.py --test`` has evaluated both models. It does not
train, choose a checkpoint or tune anything: the checkpoints were already
chosen on the validation split by train.py. Here the test predictions are
used only to pick which slices to display and to time inference.

* Visual comparison: MRI, ground truth and both models' predictions for one
  slice from each of ``--num-examples`` test series (evenly spread over the
  sorted series), using the slice with the largest ground-truth prostate area.
* Failure cases: the prostate-containing test slices where the Improved U-Net
  (the main model) has the lowest slice-level prostate Dice, at most one
  slice per series so the cases do not come from neighbouring slices. Many
  small edge slices score exactly 0, so ties go to the larger ground-truth
  prostate (the largest missed regions are shown first).
* Consistency check: the per-series test prostate Dice is recomputed with
  ``metrics.py`` and printed next to the value ``train.py`` saved in
  ``test_metrics_<checkpoint>.json``.
* Inference time: forward passes only, with the whole test split already on
  the device and one warm-up pass first; the median of ``--timing-repeats``
  passes is reported for batch size 1 and for ``--batch-size``.

Example::

    python predict.py --data-root keras_slices_data --runs-dir runs
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from dataset import CLASS_NAMES, PROSTATE_LABEL, HipMRISliceDataset
from metrics import SegmentationMetrics
from modules import build_model, count_parameters

# Model names used by train.py and the labels shown in figures.
MODELS = {"unet": "Baseline U-Net", "improved": "Improved U-Net"}
MAIN_MODEL = "improved"  # failure cases are chosen for this model
# Overlay colours for labels 1-5 (Body, Bones, Bladder, Rectum, Prostate);
# Background (0) is left transparent.
CLASS_COLOURS = ("#8d99ae", "#f4d35e", "#3a86ff", "#2a9d8f", "#e63946")
OUTLINE_COLOUR = "lime"  # ground-truth prostate outline on the prediction panels


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", default="keras_slices_data")
    parser.add_argument("--runs-dir", default="runs",
                        help="folder holding the unet/ and improved/ output folders of train.py")
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best",
                        help="best = validation-selected checkpoint (the primary result)")
    parser.add_argument("--out-dir", default=None, help="default: <runs-dir>/predict_<checkpoint>")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-examples", type=int, default=4)
    parser.add_argument("--num-failures", type=int, default=5)
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    args = parser.parse_args()
    if min(args.batch_size, args.num_examples, args.num_failures, args.timing_repeats) < 1:
        parser.error("--batch-size, --num-examples, --num-failures and --timing-repeats must be >= 1")
    return args


def load_model(name, path, device):
    """Rebuild a model and load the weights saved by train.py."""
    state = torch.load(path, map_location=device, weights_only=True)
    model = build_model(name).to(device)
    model.load_state_dict(state["model"])
    model.eval()  # dropout off; the baseline's batch norm uses its running statistics
    return model, state["epoch"]


@torch.inference_mode()
def predict_labels(model, images, device, use_amp, batch_size):
    """Return predicted label maps (N, H, W) for images of shape (N, 1, H, W)."""
    predictions = []
    for batch in images.split(batch_size):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(batch.to(device))
        predictions.append(logits.argmax(dim=1).cpu())
    return torch.cat(predictions)


@torch.inference_mode()
def time_inference(model, images, device, use_amp, batch_size, repeats):
    """Median time of forward passes over the whole split, with the data already on the device."""
    batches = images.to(device).split(batch_size)

    def one_pass():
        for batch in batches:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()  # wait for the GPU before reading the clock

    one_pass()  # warm-up: cuDNN picks its algorithms for each batch shape
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()  # after warm-up, so tuning memory is not counted
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        one_pass()
        times.append(time.perf_counter() - start)
    seconds = float(np.median(times))
    return {
        "batch_size": batch_size,
        "repeats": repeats,
        "seconds_per_pass": seconds,
        "ms_per_slice": 1000 * seconds / len(images),
        "slices_per_second": len(images) / seconds,
        # Includes the model weights and the test images held on the GPU (about 0.07 GiB).
        "peak_gpu_memory_gib": (torch.cuda.max_memory_allocated() / 2 ** 30
                                if device.type == "cuda" else None),
    }


def prostate_dice_per_slice(predictions, labels):
    """Slice-level prostate Dice (None when neither map contains prostate)."""
    pred = predictions == PROSTATE_LABEL
    target = labels == PROSTATE_LABEL
    intersection = (pred & target).sum((1, 2)).tolist()
    total = (pred.sum((1, 2)) + target.sum((1, 2))).tolist()
    return [2 * i / t if t > 0 else None for i, t in zip(intersection, total)]


def choose_examples(keys, labels, n):
    """Pick the largest ground-truth prostate slice of n evenly spread test series."""
    series = sorted({key[:2] for key in keys})
    n = min(n, len(series))
    chosen = [series[round(i * (len(series) - 1) / max(n - 1, 1))] for i in range(n)]
    areas = (labels == PROSTATE_LABEL).sum((1, 2)).tolist()
    return [max((i for i, key in enumerate(keys) if key[:2] == s), key=lambda i: areas[i])
            for s in chosen]


def choose_failures(keys, labels, dice, n):
    """Pick the lowest-Dice prostate slices of the main model, at most one per series.

    Ties (often several slices at Dice 0) go to the larger ground-truth prostate.
    """
    areas = (labels == PROSTATE_LABEL).sum((1, 2)).tolist()
    candidates = sorted((i for i in range(len(keys)) if areas[i] > 0),
                        key=lambda i: (dice[i], -areas[i]))
    picks, used_series = [], set()
    for i in candidates:
        if len(picks) == n:
            break
        if keys[i][:2] not in used_series:
            picks.append(i)
            used_series.add(keys[i][:2])
    return picks


def _format_dice(value):
    return "n/a" if value is None else f"{value:.3f}"


def save_figure(indices, images, labels, predictions, dice, keys, path, title):
    """Save one row per slice: MRI, ground truth, then each model's prediction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    cmap = ListedColormap(CLASS_COLOURS)
    columns = 2 + len(predictions)
    fig, axes = plt.subplots(len(indices), columns, figsize=(2.2 * columns, 3.9 * len(indices)),
                             squeeze=False)
    for row, index in enumerate(indices):
        image = images[index, 0].numpy()
        truth = labels[index].numpy()
        case, week, slice_index = keys[index]
        label_maps = [None, truth, *(p[index].numpy() for p in predictions.values())]
        captions = [f"Case {case} / week {week} / slice {slice_index}\n"
                    f"GT prostate {int((truth == PROSTATE_LABEL).sum())} px",
                    "Ground truth",
                    *(f"{MODELS[name]}\nprostate Dice {_format_dice(dice[name][index])}"
                      for name in predictions)]
        for column, (label_map, caption) in enumerate(zip(label_maps, captions)):
            ax = axes[row, column]
            ax.imshow(image, cmap="gray")
            if label_map is not None:
                # Labels 1-5 map onto the 5 colours; label 0 is masked out.
                ax.imshow(np.ma.masked_equal(label_map, 0), cmap=cmap, vmin=0.5, vmax=5.5,
                          alpha=0.45, interpolation="nearest")
            if column >= 2 and (truth == PROSTATE_LABEL).any():
                ax.contour(truth == PROSTATE_LABEL, levels=[0.5], colors=OUTLINE_COLOUR,
                           linewidths=0.8)
            ax.set_title(caption, fontsize=8)
            ax.axis("off")

    handles = [Patch(color=colour, alpha=0.6, label=name)
               for colour, name in zip(CLASS_COLOURS, CLASS_NAMES[1:])]
    handles.append(Line2D([], [], color=OUTLINE_COLOUR, label="ground-truth prostate outline"))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir or runs_dir / f"predict_{args.checkpoint}")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    torch.backends.cudnn.benchmark = True  # as in train.py

    # The test split with the same preprocessing as in train.py (no augmentation).
    dataset = HipMRISliceDataset(args.data_root, "test")
    samples = [dataset[i] for i in range(len(dataset))]
    images = torch.stack([s["image"] for s in samples])  # (N, 1, H, W)
    labels = torch.stack([s["label"] for s in samples])  # (N, H, W)
    keys = [(s["case"], s["week"], s["slice"]) for s in samples]

    report = {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
        "torch": str(torch.__version__), "python": platform.python_version(), "amp": use_amp,
        "checkpoint": args.checkpoint, "test_slices": len(samples), "models": {},
    }
    predictions, dice = {}, {}
    for name in MODELS:
        model_dir = runs_dir / name
        model, epoch = load_model(name, model_dir / f"{args.checkpoint}.pt", device)
        predictions[name] = predict_labels(model, images, device, use_amp, args.batch_size)
        dice[name] = prostate_dice_per_slice(predictions[name], labels)

        # Recompute the primary metric to check that the reloaded model matches train.py.
        metrics = SegmentationMetrics()
        metrics.update(predictions[name], labels, *zip(*keys))
        recomputed = metrics.summary()["prostate_dice_series_mean"]
        saved_path = model_dir / f"test_metrics_{args.checkpoint}.json"
        saved = (json.loads(saved_path.read_text())["prostate_dice_series_mean"]
                 if saved_path.is_file() else None)

        timing = [time_inference(model, images, device, use_amp, batch_size, args.timing_repeats)
                  for batch_size in (1, args.batch_size)]
        report["models"][name] = {
            "checkpoint_epoch": epoch, "parameters": count_parameters(model),
            "test_prostate_dice_recomputed": recomputed, "test_prostate_dice_saved": saved,
            "timing": timing,
        }
        print(f"{MODELS[name]} ({args.checkpoint}, epoch {epoch}): test prostate Dice "
              f"per-series mean {recomputed:.4f} (saved by train.py: "
              f"{'n/a' if saved is None else f'{saved:.4f}'})")
        for t in timing:
            print(f"  batch {t['batch_size']:2d}: {t['ms_per_slice']:.3f} ms/slice, "
                  f"{t['slices_per_second']:.0f} slices/s")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    examples = choose_examples(keys, labels, args.num_examples)
    failures = choose_failures(keys, labels, dice[MAIN_MODEL], args.num_failures)
    save_figure(examples, images, labels, predictions, dice, keys, out_dir / "comparison.png",
                f"Test slices with the largest ground-truth prostate area in each shown series "
                f"({args.checkpoint} checkpoints)")
    save_figure(failures, images, labels, predictions, dice, keys, out_dir / "failures.png",
                f"Lowest {MODELS[MAIN_MODEL]} prostate Dice on test slices, one per series "
                f"({args.checkpoint} checkpoints)")

    def describe(i):
        case, week, slice_index = keys[i]
        return {"series": f"{case}_week{week}", "slice": slice_index,
                "gt_prostate_pixels": int((labels[i] == PROSTATE_LABEL).sum()),
                "pred_prostate_pixels": {n: int((p[i] == PROSTATE_LABEL).sum())
                                         for n, p in predictions.items()},
                "prostate_dice": {n: d[i] for n, d in dice.items()}}

    report["examples"] = [describe(i) for i in examples]
    report["failures"] = [describe(i) for i in failures]
    (out_dir / "predict_summary.json").write_text(json.dumps(report, indent=2))
    print("Failure cases:")
    for f in report["failures"]:
        print(f"  {f['series']} slice {f['slice']}: GT {f['gt_prostate_pixels']} px, Dice "
              + ", ".join(f"{MODELS[n]} {_format_dice(d)}" for n, d in f["prostate_dice"].items()))
    print(f"Saved comparison.png, failures.png and predict_summary.json to {out_dir}")


if __name__ == "__main__":
    main()
