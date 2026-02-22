# Stage 1 Evaluation (Current Implementation)

This README documents the active Stage-1 evaluation pipeline in this repo.

## What is produced

- Tier2: `tier2_ppl.json`
- Tier3: `tier3_cf.json`
- Tier1 is not part of the output contract for the current launcher workflow.

## Active files and what each does

1. `evaluation/run_eval_stage1.py`

- Main orchestrator.
- Loads tokenizer with `use_fast=False` (`evaluation/run_eval_stage1.py:55`).
- Reads Tier2/Tier3 task suites from:
  - `evaluation/task_suites/stage1_ppl_slices.yaml` (`evaluation/run_eval_stage1.py:732`)
  - `evaluation/task_suites/stage1_cf_core.yaml` (`evaluation/run_eval_stage1.py:733`)
- Prepares and caches Tier2 packed tokens under `EVAL_DATA_DIR/ppl_slices/<slice>/`:
  - `packed_tokens.pt`
  - `meta.json`
    (`evaluation/run_eval_stage1.py:120`, `evaluation/run_eval_stage1.py:183`)
- Prepares and caches Tier3 task examples under `EVAL_DATA_DIR/cf_tasks/<task>/`:
  - `examples.jsonl`
  - `meta.json`
    (`evaluation/run_eval_stage1.py:400`, `evaluation/run_eval_stage1.py:440`)
- Calls external scoring backend through `LIGHTEVAL_CMD` (`evaluation/run_eval_stage1.py:682`).
- Expects backend to write:
  - `tier2_ppl.json`
  - `tier3_cf.json`
    into `.../by_checkpoint/step_<STEP>/` (`evaluation/run_eval_stage1.py:706`).

2. `evaluation/task_suites/stage1_ppl_slices.yaml`

- Defines Tier2 PPL slices and source datasets.
- Current slices:
  - `general_web` (C4 en)
  - `code` (codeparrot-clean)
  - `books_newslike` (wikitext-103-raw-v1)
  - `math_finemath3plus`
  - `math_infiwebmath3plus`
  - `web_fineweb_edu_sample10bt`
  - `wiki_en_20231101`
    (`evaluation/task_suites/stage1_ppl_slices.yaml:5`)

3. `evaluation/task_suites/stage1_cf_core.yaml`

- Defines Tier3 CF tasks and split/subset policy.
- Current tasks:
  - `hellaswag`, `piqa`, `winogrande`, `arc_easy`, `openbookqa`
  - `commonsense_qa`, `social_i_qa`, `sciq`, `qasc`, `copa`
    (`evaluation/task_suites/stage1_cf_core.yaml:5`)

4. `scripts/run_lighteval_stage1_real.sh`

- Real backend launcher used by `LIGHTEVAL_CMD`.
- Receives env vars from `run_eval_stage1.py`:
  - `LIGHTEVAL_CHECKPOINT_PATH`
  - `LIGHTEVAL_RESULTS_DIR`
  - `LIGHTEVAL_STEP`
  - `LIGHTEVAL_TOKENIZER`
  - `LIGHTEVAL_SEQ_LEN`
    (`scripts/run_lighteval_stage1_real.sh:5`)
- Calls direct backend scorer:
  - `scripts/run_lighteval_stage1_backend.py`
    (`scripts/run_lighteval_stage1_real.sh:29`)
- Sets `NANOTRON_ROOT` and runtime env, then writes:
  - `tier2_ppl.json`
  - `tier3_cf.json`

5. `scripts/run_lighteval_stage1_backend.py`

- Direct checkpoint scoring backend (no HF conversion path).
- Loads Nanotron checkpoint via Nanotron APIs (`examples.llama.convert_weights.load_nanotron_model`).
- Wraps Nanotron logits into the same interface expected by existing Tier2/Tier3 scoring helpers.
- Reuses scoring functions from `evaluation/run_eval_stage1.py`:
  - `compute_ppl` for Tier2
  - `score_candidates` for Tier3
- Writes final:
  - `tier2_ppl.json`
  - `tier3_cf.json`

6. `scripts/submit_eval_stage1.sh`

- Submission helper for fixed steps:
  - `2000 10000 20000 32000 40000 50000`
    (`scripts/submit_eval_stage1.sh:81`)
- Uses Nanotron checkpoints under:
  - `$SCRATCH/ckpts/stage1/<STEP>`
    (`scripts/submit_eval_stage1.sh:71`)
- Writes outputs under:
  - `$SCRATCH/evaluation_results/stage1`
    (`scripts/submit_eval_stage1.sh:72`)
- Validates each checkpoint has Nanotron layout:
  - `config.yaml`
  - `model_config.json`
  - `model/`
    (`scripts/submit_eval_stage1.sh:102`)
- Submits `slurm/eval_stage1_manual.sbatch` for each step.
- Uses backend command:
  - `bash scripts/run_lighteval_stage1_real.sh`
    (override with env `LIGHTEVAL_CMD`).

7. `slurm/eval_stage1_manual.sbatch`

- Per-step execution wrapper.
- Runs on GPU partition with `--gres=gpu:1` and `--cpus-per-task=8`.
- Uses `LIGHTEVAL_CMD` if exported, otherwise defaults to:
  - `bash scripts/run_lighteval_stage1_real.sh`
- Executes:
  - `python -u evaluation/run_eval_stage1.py ...`
    (`slurm/eval_stage1_manual.sbatch:81`)
- Fails job if Tier2/Tier3 outputs are missing after run (`slurm/eval_stage1_manual.sbatch:90`).

8. `scripts/stage1_prefill_cache.sh`

- CPU/internet prefill step.
- Builds Tier2/Tier3 cached input data under `EVAL_DATA_DIR` without running model scoring.
- Also ensures tokenizer/dataset assets are pulled and cached before offline GPU runs.

9. `scripts/stage1_eval_offline.sh`

- Offline submission wrapper.
- Exports:
  - `HF_DATASETS_OFFLINE=1`
  - `HF_HUB_OFFLINE=1`
  - `TRANSFORMERS_OFFLINE=1`
- Calls `scripts/submit_eval_stage1.sh` for normal per-step submission.

## Data source policy (what is online vs local)

1. Tier2/Tier3 source datasets

- Pulled from Hugging Face Hub according to YAML `hf_dataset` / `hf_config`.
- If `streaming: true`, loader tries streaming first; on failure it retries non-streaming (`evaluation/run_eval_stage1.py:101`, `evaluation/run_eval_stage1.py:110`).

2. Local cache policy

- Prepared evaluation artifacts are cached locally on shared FS at `EVAL_DATA_DIR`:
  - Tier2 packed token tensors.
  - Tier3 JSONL examples.
- Reused across checkpoints/steps once generated.

3. Model/checkpoint policy

- Input checkpoints are Nanotron directories from `$SCRATCH/ckpts/stage1/<STEP>`.
- Backend loads checkpoints directly with Nanotron and scores without HF model conversion.

## Output layout

Base:

- `$SCRATCH/evaluation_results/stage1`

Important paths:

- Prepared eval cache:
  - `$SCRATCH/evaluation_results/stage1/eval_data`
- Per-step results:
  - `$SCRATCH/evaluation_results/stage1/by_checkpoint/step_<STEP>/tier2_ppl.json`
  - `$SCRATCH/evaluation_results/stage1/by_checkpoint/step_<STEP>/tier3_cf.json`
- Slurm logs:
  - `$SCRATCH/evaluation_results/stage1/eval_logs/slurm/`

## Current final sizing

- Tier2 (PPL slices):
  - `target_tokens: 5000000` per slice
  - `target_tokens_total: 35000000` across 7 slices
- Tier3 (CF tasks):
  - `subset_size: 12000` per task (actual used can be lower if dataset split is smaller)

## Run (two-phase: prefill then offline eval)

From repo root on Leonardo:

```bash
source /leonardo_scratch/large/userexternal/$USER/venvs/nanotron_env/bin/activate
scripts/stage1_prefill_cache.sh
scripts/stage1_eval_offline.sh
```

## Monitor

```bash
squeue -u $USER
ls -lt $SCRATCH/evaluation_results/stage1/eval_logs/slurm | head
tail -f $SCRATCH/evaluation_results/stage1/eval_logs/slurm/eval_stage1_2000_<JOBID>.out
tail -f $SCRATCH/evaluation_results/stage1/eval_logs/slurm/eval_stage1_2000_<JOBID>.err
```

## Notes

- `HF_DATASETS_TRUST_REMOTE_CODE=1` is exported by launcher scripts for datasets that require remote dataset code.
- Tier2/Tier3 scoring requires GPU execution for Nanotron model loading/forward.
- Direct Nanotron scorer currently assumes checkpoint parallel layout `tp=1`, `cp=1`.

---

# Stage 2 Evaluation (HQ Tier2 + Stage1 Bridge)

Stage-2 uses the same runner/scoring flow as Stage-1:

- cache Tier2/Tier3 inputs under `EVAL_DATA_DIR`
- score with Nanotron backend via `LIGHTEVAL_CMD`
- write `tier2_ppl.json` and `tier3_cf.json` per checkpoint

Bridge policy:

- Stage-2 additionally evaluates the last Stage-1 checkpoint on the full Stage-2 suite.
- This provides a direct Stage1->Stage2 bridge baseline inside Stage-2 results.

Stage-2 goal:

- Improve Tier2 quality on Stage-2 HQ slices relative to Stage-1 bridge baseline.
- Keep Tier3 stable (no material regression) while improving Tier2.
- Bridge objective check (per step): compare against `step_50000`
  - Tier2: `delta_ppl = ppl_step - ppl_bridge` (want `<= 0`)
  - Tier3: `delta_acc = acc_step - acc_bridge` (want `>= 0`)

## Stage-2 files

1. `evaluation/run_eval_stage2.py`

- Same orchestration logic as Stage-1, but reads Stage-2 suites:
  - `evaluation/task_suites/stage2_ppl_slices.yaml`
  - `evaluation/task_suites/stage2_cf_core.yaml`

2. `evaluation/task_suites/stage2_ppl_slices.yaml`

- Keeps Stage-1 Tier2 anchor slices and adds only HQ probes:
  - `math_finemath4plus` (`HuggingFaceTB/finemath`, `finemath-4plus`)
  - `math_infiwebmath4plus` (`HuggingFaceTB/finemath`, `infiwebmath-4plus`)
  - `code_stackv2_edu_filtered` (`common-pile/stackv2_edu_filtered`)
  - `code_starcoder2data_extras_lhq` (`bigcode/starcoder2data-extras`, `lhq`)
- It does not include Stage-3 reasoning datasets.

3. `evaluation/task_suites/stage2_cf_core.yaml`

- Same Tier3 task set/policy as Stage-1 (guardrail against general-skill regression).

4. `scripts/run_lighteval_stage2_real.sh`
5. `scripts/run_lighteval_stage2_backend.py`

- Stage-2 backend launcher + Nanotron scorer.

6. `scripts/submit_eval_stage2.sh`

- Submits Stage-2 checkpoints from `$SCRATCH/ckpts/stage2`.
- Default Stage-2 checkpoints:
  - `2000 6000 10000 14000 18000 22000`
- Also submits Stage-1 bridge checkpoint by default:
  - `STEP=50000`, checkpoint path `$SCRATCH/ckpts/stage1/50000`
- Default total submissions: 7 jobs (6 Stage-2 + 1 bridge).
- Bridge can be controlled with:
  - `STAGE2_INCLUDE_STAGE1_BRIDGE=1|0`
  - `STAGE2_BRIDGE_STEP` (default `50000`)
  - `STAGE2_BRIDGE_CKPT_PATH` (default `$SCRATCH/ckpts/stage1/$STAGE2_BRIDGE_STEP`)

7. `scripts/stage2_prefill_cache.sh`
8. `scripts/stage2_eval_offline.sh`
9. `slurm/eval_stage2_manual.sbatch`

## Stage-2 run

```bash
source /leonardo_scratch/large/userexternal/$USER/venvs/nanotron_env/bin/activate
scripts/stage2_prefill_cache.sh
scripts/stage2_eval_offline.sh
```

## Stage-2 outputs

- `$SCRATCH/evaluation_results/stage2/eval_data`
- `$SCRATCH/evaluation_results/stage2/by_checkpoint/step_<STEP>/tier2_ppl.json`
- `$SCRATCH/evaluation_results/stage2/by_checkpoint/step_<STEP>/tier3_cf.json`
- Stage1->Stage2 bridge baseline appears as:
  - `$SCRATCH/evaluation_results/stage2/by_checkpoint/step_50000/`
- `$SCRATCH/evaluation_results/stage2/eval_logs/slurm/`

---

# Stage 3 Evaluation (Reasoning Additions + Stage2 Bridge)

Stage-3 keeps the same evaluator mechanics and cache/output contract:

- cache Tier2/Tier3 inputs under `EVAL_DATA_DIR`
- score with Nanotron backend via `LIGHTEVAL_CMD`
- write `tier2_ppl.json` and `tier3_cf.json` per checkpoint

Bridge policy:

- Stage-3 additionally evaluates the last Stage-2 checkpoint on the full Stage-3 suite.
- This provides a direct Stage2->Stage3 bridge baseline inside Stage-3 results.

Stage-3 goal (primary):

- Improve Tier3 (reasoning/MCQ aggregate) relative to Stage-2 bridge baseline.
- Avoid Tier2 regression relative to the same Stage-2 bridge baseline.
- In short: Tier3 gain with Tier2 no-loss.
- Bridge objective check (per step): compare against `step_22000`
  - Tier2: `delta_ppl = ppl_step - ppl_bridge` (want `<= 0`)
  - Tier3: `delta_acc = acc_step - acc_bridge` (want `>= 0`)

## Stage-3 files

1. `evaluation/run_eval_stage3.py`

- Main Stage-3 orchestrator.
- Loads:
  - `evaluation/task_suites/stage3_ppl_slices.yaml`
  - `evaluation/task_suites/stage3_cf_core.yaml`
- Includes Stage-3 CF adapters for:
  - `mmlu`
  - `arc_challenge`

2. `evaluation/task_suites/stage3_ppl_slices.yaml`

- Keeps Stage-1 anchors and Stage-2 HQ slices, and adds reasoning/QA probes:
  - `math_openmathreasoning` (`nvidia/OpenMathReasoning`, split `cot`)
  - `math_openmathinstruct1` (`nvidia/OpenMathInstruct-1`)
  - `code_opencodereasoning` (`nvidia/OpenCodeReasoning`, config/split `split_0`)

3. `evaluation/task_suites/stage3_cf_core.yaml`

- Keeps Stage-1/2 CF set, plus Stage-3 additions:
  - `mmlu` (`cais/mmlu`, config `all`)
  - `arc_challenge` (`ai2_arc`, config `ARC-Challenge`)

4. `scripts/run_lighteval_stage3_real.sh`
5. `scripts/run_lighteval_stage3_backend.py`

6. `scripts/submit_eval_stage3.sh`

- Submits Stage-3 checkpoints from `$SCRATCH/ckpts/stage3`.
- Default Stage-3 checkpoints:
  - `2000 4000 6000 8000 10000 12000`
- Also submits Stage-2 bridge checkpoint by default:
  - `STEP=22000`, checkpoint path `$SCRATCH/ckpts/stage2/22000`
- Default total submissions: 7 jobs (6 Stage-3 + 1 bridge).
- Bridge can be controlled with:
  - `STAGE3_INCLUDE_STAGE2_BRIDGE=1|0`
  - `STAGE3_BRIDGE_STEP` (default `22000`)
  - `STAGE3_BRIDGE_CKPT_PATH` (default `$SCRATCH/ckpts/stage2/$STAGE3_BRIDGE_STEP`)

7. `scripts/stage3_prefill_cache.sh`
8. `scripts/stage3_eval_offline.sh`
9. `slurm/eval_stage3_manual.sbatch`

## Stage-3 run

```bash
source /leonardo_scratch/large/userexternal/$USER/venvs/nanotron_env/bin/activate
scripts/stage3_prefill_cache.sh
scripts/stage3_eval_offline.sh
```

## Stage-3 outputs

- `$SCRATCH/evaluation_results/stage3/eval_data`
- `$SCRATCH/evaluation_results/stage3/by_checkpoint/step_<STEP>/tier2_ppl.json`
- `$SCRATCH/evaluation_results/stage3/by_checkpoint/step_<STEP>/tier3_cf.json`
- Stage2->Stage3 bridge baseline appears as:
  - `$SCRATCH/evaluation_results/stage3/by_checkpoint/step_22000/`
- `$SCRATCH/evaluation_results/stage3/eval_logs/slurm/`

---

# Stage 4 Evaluation (SFT Checkpoints + Stage3 Bridge + Tier4 SFT-native)

Stage-4 targets SFT checkpoints and now contains three scored tiers:

- Tier2: PPL slices (same Stage-3 suite)
- Tier3: CF/MCQ accuracy (same Stage-3 suite)
- Tier4: SFT-native instruction/chat behavior probes

Bridge policy:

- Stage-4 also evaluates the last Stage-3 checkpoint on the Stage-4 suite.
- This creates a direct pre-SFT vs post-SFT comparison baseline in Stage-4 outputs.

Stage-4 goal:

- Improve instruction/reasoning behavior after SFT while keeping pretraining capabilities stable.
- In this suite, that means:
  - Tier3 should improve or at least not regress materially vs the Stage-3 bridge.
  - Tier2 should remain stable (no major PPL regression) vs the Stage-3 bridge.
  - Tier4 should improve on instruction/format/chat-clean checks vs the Stage-3 bridge.

## Stage-4 files

1. `evaluation/run_eval_stage4.py`

- Main Stage-4 orchestrator.
- Reads:
  - `evaluation/task_suites/stage4_ppl_slices.yaml`
  - `evaluation/task_suites/stage4_cf_core.yaml`
  - `evaluation/task_suites/stage4_sft_native.yaml`

2. `evaluation/task_suites/stage4_ppl_slices.yaml`

- Stage-4 Tier2 suite.
- Mirrors Stage-3 slices (anchors + HQ + reasoning slices) for apples-to-apples comparison.

3. `evaluation/task_suites/stage4_cf_core.yaml`

- Stage-4 Tier3 suite.
- Mirrors Stage-3 CF tasks (including `mmlu` and `arc_challenge`).

4. `evaluation/task_suites/stage4_sft_native.yaml`

- Stage-4 Tier4 suite.
- Defines a small deterministic prompt set for SFT-native checks:
  - format following (bullets/numbered output)
  - structured output (strict JSON)
  - code output validity (Python parse check)
  - instruction priority and safety behavior
  - chat cleanliness and concise-answer behavior

5. `scripts/run_lighteval_stage4_real.sh`
6. `scripts/run_lighteval_stage4_backend.py`

- Stage-4 Nanotron scoring backend launcher and scorer.
- Backend writes:
  - `tier2_ppl.json`
  - `tier3_cf.json`
  - `tier4_sft_native.json`
  - `tier4_sft_native_generations.jsonl`

7. `scripts/submit_eval_stage4.sh`

- Submits SFT checkpoints by default from:
  - `$SCRATCH/sft_project/ckpts/sft_stage3_smoltalk_run1`
- Default behavior:
  - Uses fixed Stage-4 checkpoints:
    - `500 1500 3500 5000`
  - Also submits Stage-3 bridge checkpoint by default.
- Key overrides:
  - `STAGE4_STEPS="..."`
  - `STAGE4_INCLUDE_STAGE3_BRIDGE=1|0`
  - `STAGE4_BRIDGE_STEP` / `STAGE4_BRIDGE_CKPT_PATH`
  - `STAGE4_CONFIG_PATH`
  - `STAGE4_RESULT_ROOT`

8. `scripts/stage4_prefill_cache.sh`

- Prefills Stage-4 cache using tokenizer/sequence length from `config/sft_stage3_smoltalk.yaml`.
- Prefills all three cache blocks:
  - Tier2 packed slices
  - Tier3 CF examples
  - Tier4 prompt file

9. `scripts/stage4_eval_offline.sh`

- Offline submit wrapper for Stage-4 (`HF_*_OFFLINE=1`).

10. `slurm/eval_stage4_manual.sbatch`

- Per-step Stage-4 Slurm execution script.
- Fails the job if any tier output is missing.

## Stage-4 run

```bash
source /leonardo_scratch/large/userexternal/$USER/venvs/nanotron_env/bin/activate
scripts/stage4_prefill_cache.sh
scripts/stage4_eval_offline.sh
```

## Stage-4 outputs

- `$SCRATCH/evaluation_results/stage4/eval_data`
- `$SCRATCH/evaluation_results/stage4/by_checkpoint/step_<STEP>/tier2_ppl.json`
- `$SCRATCH/evaluation_results/stage4/by_checkpoint/step_<STEP>/tier3_cf.json`
- `$SCRATCH/evaluation_results/stage4/by_checkpoint/step_<STEP>/tier4_sft_native.json`
- `$SCRATCH/evaluation_results/stage4/by_checkpoint/step_<STEP>/tier4_sft_native_generations.jsonl`
- Stage3->Stage4 bridge baseline appears as:
  - `$SCRATCH/evaluation_results/stage4/by_checkpoint/step_<stage3_latest_or_override>/`
- `$SCRATCH/evaluation_results/stage4/eval_logs/slurm/`

## Tier4 output contract (quick reference)

`tier4_sft_native.json` contains:

- `summary`
  - `instruction_match_rate`
  - `all_checks_pass_rate`
  - `format_valid_rate`
  - `chat_clean_rate`
  - `code_parse_rate` (only for code prompts)
  - `json_valid_rate` (only for strict-json prompts)
- `by_category`
  - per-category rates for pass/format/clean
- `prompts`
  - per-prompt checks and response-level stats
