#!/bin/bash
#SBATCH --job-name=test_tta_refactor
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1             # num of GPUs
#SBATCH --mem=32G                # Memory
#SBATCH --time=01:00:00          # Walltime (HH:MM:SS)
#SBATCH --account=aip-evanesce

# Validates the TTA-method refactor (src/tta/methods/). Submit from the repo root:
#   sbatch slurm/test_tta_refactor.sh
#
#   1. static checks   - everything compiles; every duo script still parses its CLI
#                        (imports + argparse), --tta_method rejects unknown methods
#   2. self-tests      - src/tta/methods/{base,tent,eata}.py on tiny CPU models
#   3. parity          - the SAME driver (scripts/test_tta_parity.py) is run against
#                        the pre-refactor commit (git HEAD, extracted to a temp dir)
#                        and the working tree, in every duo mode on a tiny ImageNet-C
#                        stream; per-batch logits and parameter checksums must match
#                        bitwise
#   4. end-to-end      - scripts/run_dynamic_duo.py --tta_method tent through
#                        evaluate_dynamic_duo (no wandb, no plots)
#   5. eata (phase 2)  - --tta_method eata end-to-end in both_duo and both_indep, then a
#                        check that eata actually adapts (differs from no_adapt) and that
#                        no_adapt is identical under tent vs eata
# Fails (non-zero exit) if any step fails; every step runs regardless so one
# failure doesn't hide the others.

module load python/3.11.5
module load cuda/12.6
module load gcc arrow/22.0.0

source ~/py38/bin/activate

export PYTHONPATH=$PYTHONPATH:.
export TORCH_HOME=/scratch/alxstaub/torch_cache
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled

REPO=$(pwd)
CFG=cfgs/dynamic_duo_config_vitb_resnet.yaml
WORK=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/tta_refactor_XXXXXX")
fail=0
step() { echo; echo "=================== $* ==================="; }
check() { "$@" || { echo "FAILED: $*"; fail=1; }; }

step "1a. compile"
check python -m compileall -q src scripts

step "1b. every duo script still imports and parses its CLI"
for s in run_dynamic_duo compare_calibrators plot_run_diagnostics sweep_proxies proxy_vs_optimal_temperature; do
    check bash -c "python scripts/$s.py --help | grep -q -- '--tta_method'" 
    echo "  $s: ok"
done
echo "--- an unknown --tta_method must be rejected by argparse (non-zero exit expected):"
if python scripts/run_dynamic_duo.py --config $CFG --tta_method definitely_not_a_method >/dev/null 2>&1; then
    echo "FAILED: unknown --tta_method was accepted"; fail=1
else
    echo "  rejected, as expected"
fi

step "2. self-tests (tiny CPU models)"
check python -m src.tta.methods.base
check python -m src.tta.methods.tent
check python -m src.tta.methods.eata

step "3. parity: pre-refactor commit vs. working tree"
BASE=$WORK/baseline
mkdir -p $BASE
git archive HEAD | tar -x -C $BASE
# data/checkpoints are untracked/ignored; link them so the baseline tree sees the same inputs.
ln -s $REPO/data $BASE/data
ln -s $REPO/checkpoints $BASE/checkpoints
cp scripts/test_tta_parity.py $BASE/scripts/test_tta_parity.py
echo "baseline = git HEAD $(git rev-parse --short HEAD) in $BASE"
if grep -q "TTAMethod" $BASE/src/tta/dynamic_duo.py; then
    echo "FAILED: baseline already contains the refactor (HEAD was committed after it?) -- parity test is vacuous"
    fail=1
fi

echo "--- recording OLD code (twice: the second recording measures this GPU's own run-to-run noise)"
(cd $BASE && PYTHONPATH=. python scripts/test_tta_parity.py --config $CFG --out $WORK/old.pt) || { echo "FAILED: old recording"; fail=1; }
(cd $BASE && PYTHONPATH=. python scripts/test_tta_parity.py --config $CFG --out $WORK/old2.pt) || { echo "FAILED: old recording #2"; fail=1; }
echo "--- recording NEW code"
check python scripts/test_tta_parity.py --config $CFG --out $WORK/new.pt
echo "--- noise floor: old vs. old (same code twice)"
if python scripts/test_tta_parity.py --compare $WORK/old.pt $WORK/old2.pt; then
    TOL=0
    echo "GPU run is bitwise deterministic -> new vs. old must match EXACTLY"
else
    TOL=1e-3
    echo "WARNING: same code is not bitwise reproducible on this GPU; new vs. old is compared at tol=$TOL"
    echo "(a real regression -- wrong side updated, missing reset, different loss -- moves logits far more than that)"
fi
echo "--- comparing new vs. old (tol=$TOL)"
check python scripts/test_tta_parity.py --compare $WORK/old.pt $WORK/new.pt --tol $TOL
mkdir -p out/test_tta_refactor && cp $WORK/old.pt $WORK/old2.pt $WORK/new.pt out/test_tta_refactor/ 2>/dev/null

step "4. end-to-end: run_dynamic_duo.py --tta_method tent"
check python scripts/run_dynamic_duo.py --config $CFG \
    --mode both_duo --duo_calibration_mode fixed_ts --fixed_ts_config checkpoints/fixed_ts/default \
    --tta_method tent --num_samples 128 --seed 0 --no_plots

step "5. eata end-to-end (phase 2)"
EATA_COMMON="--config $CFG --duo_calibration_mode fixed_ts --fixed_ts_config checkpoints/fixed_ts/default --num_samples 128 --seed 0 --no_plots"
for mode in both_duo both_indep; do
    echo "--- eata / $mode"
    check python scripts/run_dynamic_duo.py $EATA_COMMON --mode $mode --tta_method eata > $WORK/eata_$mode.log 2>&1
    grep -E "^average" $WORK/eata_$mode.log || { echo "FAILED: no summary line for eata/$mode"; fail=1; }
done
echo "--- tent / both_indep (comparison), no_adapt under tent and under eata"
check python scripts/run_dynamic_duo.py $EATA_COMMON --mode both_indep --tta_method tent > $WORK/tent_both_indep.log 2>&1
check python scripts/run_dynamic_duo.py $EATA_COMMON --mode no_adapt --tta_method tent > $WORK/noadapt_tent.log 2>&1
check python scripts/run_dynamic_duo.py $EATA_COMMON --mode no_adapt --tta_method eata > $WORK/noadapt_eata.log 2>&1
for f in eata_both_indep tent_both_indep noadapt_tent noadapt_eata; do echo "$f: $(grep -E '^average' $WORK/$f.log)"; done
if [ "$(grep -E '^average' $WORK/noadapt_tent.log)" != "$(grep -E '^average' $WORK/noadapt_eata.log)" ]; then
    echo "FAILED: no_adapt differs between tent and eata (frozen sides must be method-independent)"; fail=1
fi
if [ "$(grep -E '^average' $WORK/eata_both_indep.log)" = "$(grep -E '^average' $WORK/noadapt_tent.log)" ]; then
    echo "FAILED: eata both_indep is identical to no_adapt (adaptation silently off?)"; fail=1
fi

echo
if [ $fail -eq 0 ]; then echo "ALL CHECKS PASSED"; else echo "SOME CHECKS FAILED (see FAILED lines above)"; fi
exit $fail
