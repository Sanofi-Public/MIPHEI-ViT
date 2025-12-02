# 🧪 Benchmark Evaluation

This folder contains all scripts and utilities to evaluate H&E → mIF prediction models, run efficiency benchmarks, generate visualizations, and compare models across datasets.

The full benchmark can be executed in a single command, and additional tools are available for inference (ROSIE), efficiency profiling, radar plots, and figure generation.

---

## 🚀 Running the Benchmark

To evaluate any model on a given dataset:

```bash
python run_benchmark.py \
    --checkpoint_dir CHECKPOINT_DIR \
    --model MODEL \
    --dataset DATASET \
    --config_dir CONFIG_DIR \
    --min_area 0
```

### Arguments

- **`-checkpoint_dir`**: Folder containing the model weights + its config file.
- **`-model`**: Model name (must match one of the evaluator name in `evaluators/model_evaluators`).
- **`-dataset`**: Dataset name (must match one of the evaluator name in `evaluators/dataset_evaluators`).
- **`-config_dir`**: Path to dataset config directory (typically `config/data/` in the root repo).
- **`-min_area`**: Minimum nuclei area (default: 0 → disables filtering, set to 10 for OrionCRC).

The evaluator automatically determines which metrics apply (pixel-level, cell-level, marker-level, etc.).

### Running ROSIE (slow models)

ROSIE inference is significantly slower than all other methods.

Therefore, **you must run inference once**, save predictions, and then run the benchmark on the saved outputs.

#### Step 1 — generate ROSIE predictions

```bash
python benchmark/scripts/rosie_inference.py \
    --checkpoint_dir CHECKPOINT_DIR \
    --pred_dir PRED_DIR \
    --dataset DATASET \
    --device cuda:0 \
    --num_workers 8
```

#### Step 2 — run the benchmark using saved predictions

```bash
python run_benchmark.py \
    --checkpoint_dir CHECKPOINT_DIR \
    --pred_dir PRED_DIR \
    --model rosie \
    --dataset DATASET \
    --config_dir config/data/
```

### Running All Evaluations (all models × all datasets)

A convenience script is provided:

```bash
bash benchmark/scripts/run_evaluations.sh
```

You may need to adapt paths depending on where checkpoints and configs are stored.

---

## 📁 Benchmark Structure

All evaluation logic is modular and located in `benchmark/evaluators`:

**1.** `BaseEvaluator` : Defines core evaluation logic and shared metrics.

2. `dataset_evaluator.py`: Contain dataset-specific logic: Define how to read *each dataset* (WSI or tiles), how to map predicted/target markers, and how to split the nuclei dataframe.

**3.** `model_evaluator.py`: Define how each model: loads weights, performs forward passes, outputs predicted mIF channels

### ➕ Adding a new dataset or model

Simply create a new evaluator in:

- `dataset_evaluators/` for datasets
- `model_evaluators/` for models

and register it.

---

## ⚡ Efficiency Benchmark

To measure inference speed, VRAM usage, throughput, and FLOPs:

```bash
python benchmark/scripts/benchmark_efficiency.py \
    --checkpoints_dir CHECKPOINTS_DIR
```

This script runs each model on a standard 256x256 tile and reports: FLOPS, peak GPU memory, model size (parameters)

---

## 🔍 Compare predicitons

To generate qualitative prediction figures (side-by-side comparisons across methods):

```bash
python benchmark/scripts/generate_figure_predictions.py \
    --checkpoint_dir CHECKPOINT_DIR \
    --output_dir OUTPUT_DIR \
    --slideindex 0 \
    --data_config config/data/orion.yaml
```

This outputs:

- H&E tile
- predicted mIF channels ( 1 image per models)
- target mIF

---

## 📊 Radar Plots (global metric summary)

You can generate radar plots summarizing metrics across models:

```bash
python visualizations/radar_plots.py \
    --checkpoints_dir CHECKPOINTS_DIR \
    --save_dir OUTPUT_DIR
```

Useful for aggregate comparison of pixel-level (Pearson, PSNR, SSIM) and cell-level (Cell AUPRC, F1 Score, ROC AUC) metrics.

---

## ⚙️ Training Scripts

All training scripts for the benchmarked models are available in `training`folder, cf `training/README.md`
