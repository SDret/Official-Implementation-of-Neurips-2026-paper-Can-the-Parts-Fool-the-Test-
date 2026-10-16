# Can the Parts Fool the Test? Counterfactual Pairing Cycles for Relational OOD

Official code for the paper **"Can the Parts Fool the Test? Counterfactual Pairing Cycles for Relational OOD"**.

A *counterfactual pairing cycle* consists of two images $u_1, u_2$ and two captions $v_1, v_2$, where
$(u_1, v_1)$ and $(u_2, v_2)$ are valid pairs and both swaps $(u_1, v_2)$, $(u_2, v_1)$ are invalid.
A score $s(u, v)$ solves the cycle if

$$\Delta = s(u_1, v_1) + s(u_2, v_2) - s(u_1, v_2) - s(u_2, v_1) > 0 .$$

By construction, every image and every caption appears equally often as positive and as negative.
Any additive score $s(u,v)=f(u)+g(v)$ therefore has $\Delta \equiv 0$, so image-only, text-only or
marker-based shortcuts score exactly chance (CycleAcc $=0.5$). Only image–text *interactions* are measured.

## Repository structure

```
.
├── cycle_oe.py            # Sec. 5.5, Table 4 / Table 8: Cycle OE vs. shortcut-contaminated OE on COCO
├── cycle_diagnostics.py   # Tables 1, 2, 3, 6, 7; Figures 1, 2; Sec. 5.1 and 5.6
└── README.md
```

| Script | Experiments |
|---|---|
| `cycle_oe.py` | Cycle OE (`softplus(-Δ)`) vs. shortcut-contaminated OE (BCE with a textual OOD marker); 4-entry BCE baseline; zero-shot CLIP; text-only / image-only probes; additive audits; marginal and validity audits |
| `cycle_diagnostics.py` | **A.** Synthetic controls (Table 1) · **B.** COCO exact-cycle diagnostics, 14 families in 4 groups (Tables 2 / 7) · **C.** Strict spatial relations, CLIP vs. rank-64 (Figure 1) · **D.** Low-rank bilinear rank sweep (Table 3 / Figure 2) · **E.** Unblocked text-marginal contamination (Sec. 5.1) · **F.** Transfer alignment on COCO (Sec. 5.6) · **G.** GQA scene-graph exact cycles and transfer (Tables 2 / 6, Sec. 5.6) |

## Requirements

- Python ≥ 3.6
- PyTorch
- transformers
- numpy
- Pillow

```bash
pip install torch transformers numpy pillow
```

A CUDA GPU is recommended. All scorers are trained on frozen CLIP features, so the GPU is mainly used for feature extraction.

## Data and models

Everything is downloaded automatically on the first run into `--data_dir` (default `./data`):

| Resource | Source |
|---|---|
| COCO 2017 instance annotations | `images.cocodataset.org` |
| COCO images (only the sampled subset) | `coco_url`, falling back to `flickr_url` |
| GQA scene graphs (`val_sceneGraphs.json`) | `downloads.cs.stanford.edu` |
| Visual Genome images (only the sampled subset) | `cs.stanford.edu/people/rak248` |
| CLIP ViT-B/32 (`openai/clip-vit-base-patch32`) | `$HF_ENDPOINT` (if set) → `hf-mirror.com` → ModelScope → `huggingface.co` |

Downloads can be resumed. If automatic download fails, place `annotations_trainval2017.zip` in
`data/coco/` and `sceneGraphs.zip` in `data/gqa/` manually.

Resulting layout:

```
data/
├── clip-vit-base-patch32/
├── coco/
│   ├── annotations/
│   ├── train2017/
│   ├── val2017/
│   ├── coco_meta.pkl         # cycle_oe.py
│   └── coco_meta_v2.pkl      # cycle_diagnostics.py
├── gqa/
│   ├── images/
│   └── gqa_val_meta.pkl
├── clip_feats.pkl            # CLIP feature cache of cycle_oe.py
└── clip_feat_store.pkl       # CLIP feature store of cycle_diagnostics.py
```

## Usage

Run the two scripts in this order:

```bash
python cycle_oe.py
python cycle_diagnostics.py
```

`cycle_diagnostics.py` imports the CLIP feature cache written by `cycle_oe.py` (`data/clip_feats.pkl`) when its own feature store is still empty.
Running the scripts in the order above reproduces the reported numbers with identical features.

Optional flags:

```bash
python cycle_diagnostics.py --skip_gqa            # COCO and synthetic experiments only
python cycle_diagnostics.py --skip_synth          # skip synthetic controls
python cycle_oe.py --seeds 0,1,2 --rank 64        # explicit defaults
```

## Arguments

### `cycle_oe.py`

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `./data` | Data / cache directory |
| `--log` | `./cycle_oe_results.log` | Output log file |
| `--seeds` | `0,1,2` | Random seeds |
| `--rank` | `64` | Rank of the bilinear interaction |
| `--epochs` | `8` | Training epochs |
| `--bs` | `512` | Batch size |
| `--lr` | `1e-3` | Learning rate (Adam) |
| `--wd` | `1e-4` | Weight decay |
| `--n_train` / `--n_val` / `--n_test` | `3000` / `500` / `1000` | Number of cycles per split |
| `--train_pool` | `6000` | Number of train2017 images (train + val pools) |
| `--val_frac` | `1/6` | Fraction of the train pool used for validation (image-disjoint) |
| `--test_pool` | `2500` | Number of val2017 images (test pool) |
| `--min_area_frac` | `0.01` | Minimum object area fraction for an object to be captioned |
| `--dl_workers` | `16` | Parallel download threads |
| `--num_workers` | `4` | DataLoader workers for CLIP encoding |

### `cycle_diagnostics.py`

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `./data` | Data / cache directory |
| `--log` | `./cycle_all_results.log` | Output log file |
| `--seeds` | `0,1,2` | Seeds for COCO / GQA experiments |
| `--synth_seeds` | `0,1,2,3,4` | Seeds for synthetic controls |
| `--epochs` | `30` | Training epochs for interaction probes |
| `--bs` | `512` | Batch size |
| `--lr` | `1e-3` | Learning rate (Adam) |
| `--wd` | `1e-4` | Weight decay |
| `--n_train` / `--n_val` / `--n_test` | `3000` / `500` / `1000` | Maximum cycles per family and split |
| `--coco_train_pool` | `10000` | Number of train2017 images |
| `--coco_test_pool` | `5000` | Number of val2017 images |
| `--gqa_pool` | `8000` | Number of GQA / Visual Genome images |
| `--skip_gqa` | off | Skip the GQA experiments |
| `--skip_synth` | off | Skip the synthetic controls |
| `--dl_workers` | `16` | Parallel download threads |
| `--num_workers` | `4` | DataLoader workers for CLIP encoding |

## Outputs

Each script prints its results and writes them to the file given by `--log`.

**`cycle_oe.py`**
- Data audit per seed:
  - number of cycles;
  - train/val image disjointness;
  - total variation between positive and negative image, caption and caption-length marginals;
  - cycle validity violations.
- Per-seed exact test CycleAcc of Cycle OE and Shortcut OE, plus their validation metrics.
- Mean ± std over seeds for:
  - Cycle OE and Shortcut OE;
  - text-only and image-only probes;
  - zero-shot CLIP;
  - the 4-entry BCE baseline.
- Additive audits on exact test cycles:
  - the additive heads of the shortcut model;
  - an additive-only probe;
  - text-only and image-only probes.

**`cycle_diagnostics.py`**
- **A.** AUROC of additive, bilinear and low-rank scorers on the four synthetic controls, including the spectrum-predicted capture per rank.
- **B.** Per-family table for 14 COCO families. Columns:
  - cycle counts;
  - marginal TV;
  - additive-audit deviation;
  - CycleAcc of zero-shot CLIP;
  - CycleAcc of diagonal and full bilinear probes;
  - CycleAcc of rank-1 / 4 / 16 / 64 probes.
  
  A per-group summary follows the table.
- **C.** Strict-relation families: zero-shot CLIP vs. rank-64.
- **D.** Rank sweep (ranks 1 / 4 / 16 / 64) on selected families.
- **E.** Unblocked AUROC vs. exact CycleAcc of text-only and additive probes trained with an OOD marker.
- **F.** Pearson / Spearman correlation between the alignment of residualized interaction matrices and cross-family transfer CycleAcc.
- **G.** The same per-family table, group summary and transfer correlation on GQA scene-graph cycles.

## Reproducibility

- **Seeding:** all randomness comes from fixed seeds:
  - image pools use seeds `12345`, `54321` and `777`;
  - cycle sampling uses `1000 + seed`, `2000 + seed` and `3000 + seed`;
  - model initialization and training are seeded per experiment.
- **Frozen encoder:** CLIP features are extracted once and cached, and all scorers are trained on these frozen features.
- **Exact evaluation:** CycleAcc is computed in float64. Ties count as 0.5, with a relative tolerance of $10^{-9}$.
- **Model selection:**
  - cycle-trained models use held-out validation CycleAcc;
  - pair-trained models use validation AUROC.
- **Image availability:** COCO / Visual Genome images are fetched from their public servers. Images that are unavailable are dropped, so the image pools depend on what the servers return.

## Citation

```bibtex
@inproceedings{counterfactual_pairing_cycles,
  title     = {Can the Parts Fool the Test? Counterfactual Pairing Cycles for Relational OOD},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025}
}
```

## License

This code is released under the MIT License. COCO, GQA, Visual Genome and CLIP are subject to their own licenses and terms of use.
