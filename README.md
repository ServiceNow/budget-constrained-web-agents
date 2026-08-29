# 💸 Budget-Constrained Web Agents

This repository contains the code for the paper
[**Are Online Skill and Memory Modules Always Worth Their Tokens? A Budget-Constrained Study of Web Agents**](https://arxiv.org/abs/2606.15017).

> Online web agents often augment a base actor with memory, workflow, or skill modules. These modules can improve performance, but they also consume test-time tokens, a cost rarely reported alongside the actor's inference cost. We study online augmentation, where this overhead is paid on every task, and re-evaluate its benefits under a fixed total inference budget. We compare AWM, ASI, and ReasoningBank with a token-matched vanilla baseline that uses the same budget for additional actor steps. Across four WebArena domains and three models, Gemini 3 Flash, GPT-5.4-mini, and Qwen 3.6-27B, the vanilla baseline matches or surpasses all three augmentation methods in aggregate success rate while often using fewer total tokens. We observe a similar trend on WorkArena-L1 with Qwen 3.6-27B, indicating that the effect extends to enterprise knowledge-work tasks. Our results suggest that skills and workflow memory can be useful in specific domains, but their apparent gains often vanish against a budget-matched actor. We further show that run-to-run variance materially affects outcomes and should be reported as a core evaluation criterion for online web agents.

## 🛠️ Setup

Install the following in your environment (we used `python==3.12`):

```bash
pip install -r requirements.txt
pip install --no-deps libwebarena==0.0.5
playwright install chromium
```

**Environment variables.**
Copy the example environment file to `.env` and fill in your provider credentials and site URLs.
Set any site you are not hosting to `todo` to skip tasks that need it. 

**Generate task configs (AWM / ASI only).**
The **AWM** and **ASI** pipelines auto-evaluate and induce from solved trajectories, which requires per-task config files.
Generate the configs into the run id you will use (`default` here, see the **How runs are organized** section below):

```bash
python constbudg/config_files/generate_test_data.py \
    --input constbudg/config_files/test.raw.json \
    --output-dir constbudg/config_files/default/webarena
```

## 🚀 Running experiments

The pipeline is driven by `constbudg/run_online.py`, run from inside the package directory (`constbudg/`).
The `--experiment` flag selects the configuration, and the budget knobs differ per configuration:

- **AWM / ASI**: `--experiment awm` or `--experiment asi`, with `--max_steps 10`.
- **Vanilla-IB** (the budget-matched baseline): `--experiment vanilla`, with `--max_steps 15` and `--prune_axtree`.

### 🔖 How runs are organized: `--run_suffix`

Every run has a **run id** (`--run_suffix`, default `default`). It scopes *all* of a run's artifacts under that id, so independent runs never overwrite each other:

```
constbudg/results/<run_suffix>/        # trajectories, rewards
constbudg/config_files/<run_suffix>/   # generated task configs
constbudg/actions/<run_suffix>/        # induced programmatic skills (ASI)
constbudg/workflows/<run_suffix>/      # induced workflows (AWM)
constbudg/outputs/<run_suffix>/        # induction intermediates
```

Give each experiment its own suffix and its induced skill/workflow library stays isolated to that experiment. If you generate task configs (above), use the **same** suffix.

### ▶️ Example commands

Running the full WebArena shopping set with Gemini 3 Flash:

```bash
# ASI
python run_online.py --experiment asi --website shopping \
    --model_name openrouter/google/gemini-3-flash-preview \
    --max_steps 10 --run_suffix run1

# AWM
python run_online.py --experiment awm --website shopping \
    --model_name openrouter/google/gemini-3-flash-preview \
    --max_steps 10 --run_suffix run2

# Vanilla-IB (budget-matched baseline)
python run_online.py --experiment vanilla --website shopping \
    --model_name openrouter/google/gemini-3-flash-preview \
    --max_steps 15 --prune_axtree --run_suffix run3
```

### 🧰 The `run.sh` launcher (optional)

`run.sh` at the repo root is a convenience wrapper. It assumes your sites are already hosted and site URLs are set in `.env`, then: loads `.env`, optionally (re)generates task configs and runs a short throwaway warm-up so the first real task isn't slowed by cold browser/site startup, runs `run_online.py` while teeing all output to a log file, and on success, archives the run's artifacts to `warehouse/<log-name>/` with a `stats.txt` summary.

```bash
# ./run.sh <log_file> [run_online.py args...]

# Vanilla-IB
./run.sh vanilla_ib_shopping_flash3.log --experiment vanilla --website shopping \
    --model_name openrouter/google/gemini-3-flash-preview \
    --max_steps 15 --prune_axtree --run_suffix run1

# ASI
./run.sh asi_shopping_flash3.log --experiment asi --website shopping \
    --model_name openrouter/google/gemini-3-flash-preview \
    --max_steps 10 --run_suffix run2
```

Behavior is tunable with environment variables: `CLEANUP=no|delete|move`, `GENCONFIG=1|0`, `WARMUP=1|0`, `STATS=1|0`.

## 🩹 Troubleshooting

**Playwright timeouts.** WebArena sites on slow backends can exceed BrowserGym's default 500 ms action timeout (e.g., `TimeoutError: Locator.click: Timeout 500ms exceeded`). BrowserGym does not expose this timeout as a parameter and it is hardcoded, so the only way to raise it is to edit the value in the installed package:

```bash
BG_ACTION="<YOUR ENVIRONMENT PATH>/lib/python3.12/site-packages/browsergym/core/action"
sed -i 's/timeout=500/timeout=4000/g' "$BG_ACTION/functions.py"
sed -i 's/timeout=500/timeout=4000/g' "$BG_ACTION/utils.py"
```

## 🙏 Acknowledgements

This codebase is based on the [Agent Skill Induction (ASI)](https://github.com/zorazrw/agent-skill-induction) repository, and also draws on ideas from [ReasoningBank](https://github.com/google-research/reasoning-bank). It is built on top of [BrowserGym](https://github.com/ServiceNow/BrowserGym) for browser-based agent environments. Many thanks to the authors of all these projects.

## 📚 Citation

If you use this code, please cite:

```bibtex
@inproceedings{hajimiri2026budget,
  title     = {Are Online Skill and Memory Modules Always Worth Their Tokens? {A} Budget-Constrained Study of Web Agents},
  author    = {Hajimiri, Sina and Aminbeidokhti, Masih and Dolz, Jose and Ben Ayed, Ismail and Laradji, Issam H. and Gella, Spandana and Gontier, Nicolas},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026},
}
```

## 📄 License

Released under the license in [`LICENSE`](LICENSE) (Creative Commons Attribution-ShareAlike 4.0 International, CC BY-SA 4.0), matching the upstream ASI repository.
