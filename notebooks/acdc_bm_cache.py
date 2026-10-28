"""Build the BM softmax cache that notebooks/acdc_periodic_verification.py reads with UTRACE_NB_MODEL=bm.

BM is the Bai et al. (2018) fully convolutional network of `ukbb_cardiac`, a TensorFlow 1 checkpoint
(FCN_sa.{meta,index,data-*}). TensorFlow has no wheel for the project's Python, so this script runs in
an isolated environment, never in the project's (see CONTRIBUTING.md, "Notebooks", for the command).

It repeats the notebook's loader (patients of training/ and testing/, never validation/; crop/pad to
256 x 256; relabel; the foreground pixel selection; the per-slice degradation seeds) and writes
`acdc_bm_fg_<level>.npz` and `acdc_bm_fp_<level>.npz` in the notebook's cache format. Where KM's cache
file of the same level exists, the labels, patients and slices of the foreground file are asserted equal
to it: only the softmax differs.

BM's preprocessing follows `ukbb_cardiac/common/deploy_network.py` and `image_utils.rescale_intensity`:
each 3-D volume (here one ED or ES frame; the original rescales the whole 4-D sequence) is clipped to
its 1st-99th percentiles and mapped to [0, 1]; the network is fed at the native resolution (no
resampling), the image padded with zeros to a multiple of 16. Here the 256 x 256 crop/pad of the
notebook (which is a multiple of 16) is applied AFTER the rescaling, so the padding is zero in the
network's input scale, as in the original. The degradations act on that rescaled input, as they act on
KM's min-max input (the same code and the same seeds): noise is added to it, blur and bias field are
applied to it and followed by the per-slice min-max, exactly as for KM. Level 0 is the rescaled input
itself.

The files are written as uncompressed .npz with a fixed timestamp, so that two builds are byte-identical.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import nibabel as nib
import numpy as np
import tensorflow.compat.v1 as tf
import torch
import torch.nn.functional as F
from monai.transforms import GaussianSmooth, RandBiasField

SEED = 20261112                       # the notebook's seed
BATCH = 32                            # slices per forward pass, within one patient (the notebook's)
# The full-mode level grids of the notebook (all levels its cache needs), plus any BM-only extras from --extra.
LEVELS = {"awgn": [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75],
          "blur": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
          "bias": [0.25, 0.5]}
DEG_ID = {"clean": 0, "awgn": 1, "blur": 2, "bias": 3}
HW = 256 * 256


def find_repo_root() -> Path:
    for q in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (q / "pyproject.toml").exists() and (q / "src" / "utrace").exists():
            return q
    raise RuntimeError("run this script from inside the utrace repository")


def resize_and_pad(t: torch.Tensor) -> torch.Tensor:
    """utrace's ACDCDataset.resize_and_pad to 256 x 256: centre crop if larger, zeros at the bottom/right if smaller."""
    _, h, w = t.shape
    if h > 256:
        s = (h - 256) // 2
        t = t[:, s:s + 256, :]
    if w > 256:
        s = (w - 256) // 2
        t = t[:, :, s:s + 256]
    return F.pad(t, (0, max(0, 256 - t.shape[2]), 0, max(0, 256 - t.shape[1])), mode="constant", value=0)


def relabel(m: torch.Tensor) -> torch.Tensor:
    """utrace.utils.utils.relabel: swaps ACDC's 1 (RV) and 3 (LV)."""
    m[m == 3] = 10
    m[m == 1] = 3
    m[m == 10] = 1
    return m


def rescale_intensity(image: np.ndarray, thres=(1.0, 99.0)) -> np.ndarray:
    """ukbb_cardiac common/image_utils.py: clip to the percentiles, map to [0, 1] (not in place here)."""
    val_l, val_h = np.percentile(image, thres)
    image2 = np.clip(image, val_l, val_h)
    return ((image2.astype(np.float32) - val_l) / (val_h - val_l)).astype(np.float32)


def minmax(x: torch.Tensor) -> torch.Tensor:
    lo = x.amin(dim=(1, 2, 3), keepdim=True)
    hi = x.amax(dim=(1, 2, 3), keepdim=True)
    return (x - lo) / (hi - lo)


def slice_seed(deg: str, level: float, pi: int, si: int) -> int:
    return int(np.random.SeedSequence([SEED, DEG_ID[deg], round(level * 1000), pi, si]).generate_state(1)[0])


def _plain(y) -> torch.Tensor:
    y = y.as_tensor() if hasattr(y, "as_tensor") else torch.as_tensor(y)
    return y.to(torch.float32)


def degrade(x_in: torch.Tensor, deg: str, level: float, pi: int) -> torch.Tensor:
    """The notebook's `degrade`, with BM's rescaled input in place of KM's min-max input (see the module docstring)."""
    x = x_in
    if level == 0.0:
        return x
    out = []
    for si in range(len(x)):
        seed = slice_seed(deg, level, pi, si)
        if deg == "awgn":
            g = torch.Generator().manual_seed(seed)
            out.append(x[si] + torch.randn(x[si].shape, generator=g) * level)
        elif deg == "blur":
            out.append(_plain(GaussianSmooth(sigma=level)(x[si].clone())))
        elif deg == "bias":
            t = RandBiasField(degree=3, coeff_range=(-level, level), prob=1.0)
            t.set_random_state(seed=seed)
            out.append(_plain(t(x[si].clone())))
        else:
            raise ValueError(deg)
    xs = torch.stack(out)
    return xs if deg == "awgn" else minmax(xs)


def write_npz(f: Path, **arrays):
    """np.savez-compatible, uncompressed, with a fixed timestamp, so that the bytes depend only on the arrays."""
    tmp = f.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for name, a in arrays.items():
            zi = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_STORED
            with z.open(zi, "w", force_zip64=True) as fh:
                np.lib.format.write_array(fh, np.ascontiguousarray(a), allow_pickle=False)
    tmp.rename(f)


def file_sha256(f: Path) -> str:
    return hashlib.sha256(f.read_bytes()).hexdigest()


def main():
    repo = find_repo_root()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=os.environ.get("UTRACE_ACDC_DATA", str(repo / "scripts" / "../../data/ACDC")))
    ap.add_argument("--model", default=os.environ.get("UTRACE_ACDC_BM", str(repo / "scripts" / "../../models/kaggle")),
                    help="directory holding FCN_sa.{meta,index,data-*}")
    ap.add_argument("--out", default=str(repo / "notebooks" / "_build" / "cache"))
    ap.add_argument("--extra", default="awgn:0.005,awgn:0.01,awgn:0.02,awgn:0.03",
                    help="BM-only levels added to the notebook's grid (BM collapses at the smallest common noise levels)")
    ap.add_argument("--overlay-patient", default="patient095",
                    help="patient whose whole softmax and model input are also written (acdc_bm_ov_*), for the notebook's overlay")
    ap.add_argument("--only", default="", help="build only these levels, e.g. clean,awgn:0.5 (a check)")
    a = ap.parse_args()
    data, mdir, out = Path(a.data).resolve(), Path(a.model).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    levels = {d: list(v) for d, v in LEVELS.items()}
    for item in filter(None, a.extra.split(",")):
        d, lv = item.split(":")
        levels[d] = sorted(set(levels[d]) | {float(lv)})
    keys = [("clean", 0.0)] + [(d, lv) for d in ("awgn", "blur", "bias") for lv in levels[d]]
    if a.only:
        keep = {(k.split(":")[0], float(k.split(":")[1]) if ":" in k else 0.0) for k in a.only.split(",")}
        keys = [k for k in keys if k in keep]

    # --- patients: training/ and testing/ only (the notebook's loader) ---
    patients = []
    for subset in ("training", "testing"):
        for pdir in sorted((data / subset).glob("patient*")):
            info = dict(line.split(": ", 1) for line in (pdir / "Info.cfg").read_text().strip().splitlines())
            vols = []
            for gt in sorted(pdir.glob("*_gt.nii.gz")):
                frame = int(gt.name.split("frame")[1][:2])
                assert frame in (int(info["ED"]), int(info["ES"])), (pdir.name, frame)
                vols.append({"img": str(gt).replace("_gt.nii.gz", ".nii.gz"), "gt": str(gt)})
            assert len(vols) == 2, pdir.name
            patients.append({"id": pdir.name, "vols": vols})
    assert len(patients) == 150

    xs, ys, pat_range = [], [], []
    for p in patients:
        a0 = len(xs)
        for v in p["vols"]:
            img = nib.load(v["img"]).get_fdata()
            lab = nib.load(v["gt"]).get_fdata()
            img_bm = rescale_intensity(img)                      # per volume, on the native image, before crop/pad
            for s in range(lab.shape[2]):
                xs.append(resize_and_pad(torch.tensor(img_bm[:, :, s], dtype=torch.float32).unsqueeze(0)))
                ys.append(relabel(resize_and_pad(torch.tensor(lab[:, :, s], dtype=torch.int).unsqueeze(0))))
        pat_range.append((a0, len(xs)))
    x_bm = torch.stack(xs)
    y_all = torch.stack(ys)[:, 0].numpy().astype(np.int8)
    del xs, ys
    n_slices = len(x_bm)
    slice_pat = np.concatenate([np.full(b - a0, i) for i, (a0, b) in enumerate(pat_range)])
    fg_flat = y_all.reshape(-1) > 0
    fg_y = y_all.reshape(-1)[fg_flat].astype(np.int8)
    fg_p = np.repeat(slice_pat, HW)[fg_flat].astype(np.int16)
    fg_s = np.repeat(np.arange(n_slices), HW)[fg_flat].astype(np.int16)
    print(f"{len(patients)} patients, {n_slices} slices, {len(fg_y):,} foreground pixels; BM input range "
          f"[{x_bm.min():.3f}, {x_bm.max():.3f}]", flush=True)

    # --- BM: the checkpoint as misas and ukbb_cardiac load it ---
    tf.disable_eager_execution()
    sess = tf.Session()
    saver = tf.train.import_meta_graph(str(mdir / "FCN_sa.meta"))
    saver.restore(sess, str(mdir / "FCN_sa"))
    g = tf.get_default_graph()
    assert g.get_tensor_by_name("image:0").shape.as_list() == [None, None, None, 1]
    assert g.get_tensor_by_name("prob:0").shape.as_list() == [None, None, None, 4]

    def infer(x: torch.Tensor) -> np.ndarray:
        """(S, 1, 256, 256) -> softmax (S * 65536, 4) float32, slice order then row-major, in batches of BATCH."""
        parts = []
        for s0 in range(0, len(x), BATCH):
            xb = x[s0:s0 + BATCH, 0].numpy()[..., None].astype(np.float32)
            prob = sess.run("prob:0", feed_dict={"image:0": xb, "training:0": False})
            assert prob.shape == (len(xb), 256, 256, 4)
            parts.append(prob.reshape(-1, 4))
        return np.concatenate(parts)

    meta_path = out / "acdc_bm_cache_meta.json"
    old = json.loads(meta_path.read_text()) if meta_path.exists() else {"levels": {}}   # levels built earlier are kept
    meta = {"tensorflow": tf.__version__, "torch": torch.__version__, "numpy": np.__version__,
            "nibabel": nib.__version__, "python": sys.version.split()[0],
            "checkpoint_sha256": {p.name: file_sha256(p) for p in sorted(mdir.glob("FCN_sa.*"))},
            "preprocessing": "ukbb_cardiac rescale_intensity (1, 99) per 3-D volume, then crop/pad to 256x256 (zeros)",
            "levels": old["levels"], **{k: v for k, v in old.items() if k == "clean_confusion_fg"}}
    for key in keys:
        t0 = time.time()
        f = out / ("acdc_bm_fg_clean.npz" if key[0] == "clean" else f"acdc_bm_fg_{key[0]}_{key[1]:g}.npz")
        f_fp = f.with_name(f.name.replace("_fg_", "_fp_"))
        parts, fp, fp_pat, fp_sl = [], [], [], []
        for pi, (a0, b) in enumerate(pat_range):
            x = degrade(x_bm[a0:b], key[0], key[1], pi)
            pr = infer(x)
            if patients[pi]["id"] == a.overlay_patient:
                f_ov = f.with_name(f.name.replace("_fg_", "_ov_"))
                write_npz(f_ov, smx=pr, x=x[:, 0].numpy())
            fg = fg_flat[a0 * HW:b * HW]
            parts.append(pr[fg])
            m = ~fg & (pr.argmax(1) > 0)
            fp.append(pr[m])
            fp_pat.append(np.full(m.sum(), pi, np.int16))
            fp_sl.append(np.repeat(np.arange(a0, b, dtype=np.int16), HW)[m])
        smx = np.concatenate(parts)
        assert smx.dtype == np.float32 and smx.shape == (len(fg_y), 4)
        km = out / f.name.replace("acdc_bm_", "acdc_km_")
        if not km.exists():                                       # a BM-only level: the selection does not depend on the level
            km = out / "acdc_km_fg_clean.npz"
        if km.exists():                                           # only the softmax may differ from KM's file
            z = np.load(km)
            assert np.array_equal(z["y"], fg_y) and np.array_equal(z["patient"], fg_p) and np.array_equal(z["slice"], fg_s), \
                f"labels, patients or slices differ from {km.name}"
            km_checked = True
        else:
            km_checked = False
        write_npz(f, smx=smx, y=fg_y, patient=fg_p, slice=fg_s)
        write_npz(f_fp, smx=np.concatenate(fp), patient=np.concatenate(fp_pat), slice=np.concatenate(fp_sl))
        if key == ("clean", 0.0):                                 # BM's class order is 0 BG, 1 LV, 2 MYO, 3 RV, like KM's
            conf = np.zeros((4, 4), np.int64)                     # rows: true label (1..3), columns: BM's argmax
            np.add.at(conf, (fg_y.astype(int), smx.argmax(1)), 1)
            for k in (1, 2, 3):
                assert conf[k].argmax() == k and conf[:, k][1:].argmax() == k - 1, f"class order: {conf.tolist()}"
            meta["clean_confusion_fg"] = conf.tolist()
        meta["levels"][f"{key[0]}_{key[1]:g}"] = {"seconds": time.time() - t0, "labels_equal_km_cache": km_checked,
                                                    "n_false_positive": int(sum(len(q) for q in fp)),
                                                    "sha256": {f.name: file_sha256(f), f_fp.name: file_sha256(f_fp),
                                                               f_ov.name: file_sha256(f_ov)}}
        print(f"{key}: {time.time() - t0:.0f}s, KM labels {'equal' if km_checked else 'not checked'}", flush=True)
    meta["seconds_total"] = meta.get("seconds_total", 0.0) + time.time() - t_start
    meta_path.write_text(json.dumps(meta, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
