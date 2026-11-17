# Evaluating the linear-decay k-MIP model at fixed k (k=2, k=15, …)

**Model:** `GPS+kmip-linear-50-to-10-linear` on PascalVOC-SP. k decayed linearly from 50 to 10 over 200 epochs. There are 4 seeds.
**Goal:** keep the trained weights frozen, set the same k on all 8 SparseAttention layers, and measure val/test F1.

## What's in the tarball (checked)

| seed | best ckpt epoch | k the schedule used at that epoch |
|---|---|---|
| 0 | 108 | 28 |
| 1 | 151 | 20 |
| 2 | 140 | 22 |
| 3 | 147 | 20 |

* The saved checkpoint is the **best-val** epoch, not the last one. So the model you're evaluating was trained with k≈20–28 at that point, not k=10.
* `config.yaml` says hidden dim 68, but training auto-scaled it to **72** (the training log says so). `eval_fixed_k.py` reads the dim from the checkpoint to handle this.
* Don't use `main.py` with `pretrained.dir` for this. `load_pretrained_model_cfg` copies `cfg.gt` from the saved config, which silently resets k to the schedule's start (k=50). The script sets k directly on the model instead.

---

## Step 1: Open Linux on the Windows PC (WSL)

The training ran in WSL on a Windows desktop (`/home/shaikha/...` on `DESKTOP-BMP8B26`), so use WSL here too. In the remote session, open **PowerShell**:

```powershell
wsl -l -v
```

* If Ubuntu is listed, open it: `wsl -d Ubuntu`
* If nothing is listed, run `wsl --install -d Ubuntu`, reboot, then open Ubuntu from the Start menu.

**Shortcut:** if this is the same PC the model was trained on, check whether the repo is already there:
```bash
ls ~/adaptive-k-MIP-GraphGPS /home/*/adaptive-k-MIP-GraphGPS 2>/dev/null
```
If it is there (with its `.venv` and `datasets/`), `cd` into it, run `source .venv/bin/activate`, then skip to **Step 4**.

## Step 2: Get the code

```bash
cd ~
git clone https://github.com/ZoyaAnwar/adaptive-k-MIP-GraphGPS.git
cd adaptive-k-MIP-GraphGPS
```
(If `eval_fixed_k.py` was pushed to the repo, it's already here. Otherwise see Step 4.)

## Step 3: Python environment

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-mac.txt          # works on Linux too
python -c "import torch;print(torch.__version__, torch.cuda.is_available())"
```
Then install torch-scatter for your torch version. Use `cu121` if CUDA printed True, `cpu` otherwise:
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
```
(If `rdkit` fails to install, remove it from the requirements file. PascalVOC doesn't need it.)

## Step 4: Copy the tarball and script over

In the **Windows App** remote session, copy `kmip-results 2.tar.gz` and `eval_fixed_k.py` from your Mac and paste them into Windows `Downloads`. Copy/paste works through the remote session's clipboard. Then in WSL (replace `<WinUser>`):

```bash
cd ~/adaptive-k-MIP-GraphGPS
cp "/mnt/c/Users/<WinUser>/Downloads/kmip-results 2.tar.gz" .
cp "/mnt/c/Users/<WinUser>/Downloads/eval_fixed_k.py" .      # skip if it came via git
tar -xzf "kmip-results 2.tar.gz"          # creates results/local/PascalVOC-SP/...
ls results/local/PascalVOC-SP/full/linear/GPS+kmip-linear-50-to-10-linear
```
You should see `config.yaml  0  1  2  3  agg`.

## Step 5: Sanity check (no dataset needed, ~10 s)

```bash
RUN="results/local/PascalVOC-SP/full/linear/GPS+kmip-linear-50-to-10-linear"
python eval_fixed_k.py --run-dir "$RUN" --ks 2 15 --dry-run
```
Expect `hidden 72, params 480469` for all 4 seeds, then `Dry run OK.`

## Step 6: Run the evaluation

```bash
python eval_fixed_k.py --run-dir "$RUN" --ks 2 15 --splits val test
```
* The **first run downloads and processes PascalVOC-SP** into `./datasets` (a few GB, can take a while). Later runs reuse it.
* For each seed, the script also evaluates the k the schedule used at that checkpoint (`trained_k`) as a baseline.
* Add `--device cpu` if CUDA isn't working. Eval is light either way.

Recommended, a full sweep (gives you a k-vs-F1 curve):
```bash
python eval_fixed_k.py --run-dir "$RUN" --ks 1 2 5 10 15 20 30 50 100
```

## Step 7: Read the results

Results are written to `$RUN/fixed_k_eval/`:
* `summary.csv`: mean ± std over the 4 seeds for each k and split (f1, accuracy, mIoU, auc, loss)
* `fixed_k_results.csv`: one row per seed/k/split
* `eval.log`: full console output

```bash
column -s, -t < "$RUN/fixed_k_eval/summary.csv" | cut -c1-120
```

**What to compare against**
* `trained` row: the same weights at the k they were trained with (≈20–28). This is your reference.
* `k=10`: the schedule's end value.
* The original **fixed-k=15** model: its numbers are already in `results/local/PascalVOC-SP/full/fixed/GPS+kmip-fixed/*/test/stats.json`. This answers whether the linear-decay model at k=15 beats or matches a model trained at k=15 the whole time.
* Expect k=2 to drop noticeably (very little global attention). If F1 stays flat from 10 to 50, the model is robust to k at inference, which supports the curriculum.

## Step 8 (optional): Bring results back / commit

```bash
git checkout -b fixed-k-eval
git add eval_fixed_k.py "$RUN/fixed_k_eval/summary.csv" "$RUN/fixed_k_eval/fixed_k_results.csv"
git commit -m "Fixed-k evaluation of linear-decay PascalVOC model"
git push -u origin fixed-k-eval
```
(Don't commit the `.ckpt` files or `datasets/`.)

## Troubleshooting

| Error | Fix |
|---|---|
| `size mismatch ... 72 vs 68` | You're using an old copy of the script. Use the version that reads the dim from the checkpoint. |
| `No module named torch_scatter` | Re-run the torch-scatter install with the matching torch/CUDA tag. |
| Killed / out of memory during dataset processing | Raise WSL memory in `C:\Users\<WinUser>\.wslconfig` (`[wsl2]` then `memory=16GB` on its own line), then run `wsl --shutdown`. |
| `CUDA not available` | Add `--device cpu`, or install the NVIDIA Windows driver (WSL picks it up automatically). |
