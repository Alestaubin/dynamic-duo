#!/bin/bash
#SBATCH --job-name=paper_tables
#SBATCH --output=logs/%j_paper_tables.out
#SBATCH --error=logs/%j_paper_tables.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1             # num of GPUs
#SBATCH --mem=32G                # Memory
#SBATCH --time=10:00:00          # Walltime (HH:MM:SS)
#SBATCH --account=aip-evanesce

# Regenerates every paper table for ViT-B/16 + ResNet-50 on all 15 ImageNet-C corruptions
# (zoom_blur added to EVAL). Submit from the repo root: sbatch slurm/run_paper_tables.sh
#
# Step 1 is the only expensive part (both models adapt under Tent, logits cached). Under
# both_indep the calibrator never affects adaptation, so every other calibrator table is a
# zero-forward-pass replay of those cached logits (step 3). Re-running after a timeout in
# step 3: comment out step 1 and re-submit, the cached logits are reused.

module load python/3.11.5
module load cuda/12.6
module load gcc arrow/22.0.0

source ~/py38/bin/activate

export PYTHONPATH=$PYTHONPATH:.
export TORCH_HOME=/scratch/alxstaub/torch_cache
export PYTHONUNBUFFERED=1

set -e

CFG=cfgs/dynamic_duo_config_vitb_resnet.yaml
RUN_NAME=paper_vitb_resnet_both_indep
RUN=out/run_diagnostics/$RUN_NAME
T=out/paper_tables
mkdir -p $T

# 1) Adapting run: main duo row (beta=0.5, pbs=128, log pool) + Table 1 sharpness block live.
#    Also gives the Adapting members and Adapting duo rows.
python scripts/plot_run_diagnostics.p
    --calib_config cfgs/calib_configs/nuclear_norm_identity_pbs128.json \
    --mode both_indep --num_samples 50000 --seed 0 --cache_logits \
    --run_name $RUN_NAME \
    --compare_configs cfgs/compare_runs/paper_beta_log_pbs128.json
cp $RUN/accuracy_table.tex $T/t1_beta_log_acc.tex
cp $RUN/ece_table.tex      $T/t1_beta_log_ece.tex

# 3) Replays from cached logits. Each replay REPLACES the compare set and overwrites
#    accuracy_table.tex / ece_table.tex, so the tables are copied out after each one.
replay () {  # $1 = compare file stem, $2 = output tag
    python scripts/plot_run_diagnostics.py --config $CFG --csv_dir $RUN \
        --compare_configs cfgs/compare_runs/$1.json
    cp $RUN/accuracy_table.tex $T/$2_acc.tex
    cp $RUN/ece_table.tex      $T/$2_ece.tex
}
replay paper_pbs_log_beta0.5      t1_pbs_log          # Table 1, proxy-batch-size block
replay paper_pool_beta0.5_pbs128  t4_t5_pool          # Tables 4 and 5 (Table 3 is extracted from these)
replay paper_beta_linear_pbs256   t6_t7_beta_linear   # Tables 6 and 7
replay paper_pbs_linear_beta0.5   t8_pbs_linear       # Table 8
replay paper_ema_alpha            t2_ema              # Table 2 (appendix A)
replay paper_beta_log_pbs128      fig1                # last: gate_weight_*.png (Fig. 1) shows every beta

# 2) Frozen rows of Table 1 (ResNet-50, ViT-B/16, Duo fixed TS): no adaptation.
python scripts/plot_run_diagnostics.py --config $CFG \
    --calib_config cfgs/calib_configs/fixed_ts_default.json \
    --mode no_adapt --num_samples 50000 --seed 0 \
    --run_name paper_vitb_resnet_frozen
cp out/run_diagnostics/paper_vitb_resnet_frozen/accuracy_table.tex $T/t1_frozen_acc.tex

echo "Done. Tables in $T, Fig. 1 plots in $RUN/per_corruption/gate_weight_{brightness,fog}.png"
