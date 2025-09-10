# AGENTS.md

> **Audience:** AI coding agent (Codex) and contributors working on **pi0-edit**  
> **Goal:** Edit **π₀** action trajectories on top of the **pi0-text-latent** baseline.  
> **Key fact:** **Env setup = pi0-text-latent README**, **but run (server/client) = pi0-edit style (two terminals)**.

---

## 1) Project scope & ground rules

- We modify/extend **action-trajectory editing** for π₀.  
- We **reuse the baseline environment** from `pi0-text-latent`.  
- **Running topology is two-process**: one **server** and one **client** communicating via HTTP on a fixed port.  
- **Do not** collapse the two environments into one; we intentionally keep:
  - **Conda env** for policy server (`text-latent`, Python 3.11)
  - **LIBERO venv** for the client (`examples/libero/.venv`, Python 3.8)
- Keep commands **idempotent** and **non-interactive**; any server-side code change **requires a server restart**.

---

## 2) Environments

Follow the official baseline (pi0-text-latent) installation.

### 2.1 Policy env (server)
```bash
conda create -n text-latent python=3.11 -y
conda activate text-latent
pip install uv
cd pi0-text-latent/
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### 2.2 LIBERO env (client)

```bash
cd pi0-text-latent/
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/modified_libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/modified_libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/modified_libero
```

### 2.3 Optional: ready-made text-latent (exp\_data)

```bash
cd pi0-text-latent/
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/Shady0057/pi0-text-latent
mv pi0-text-latent exp_data
cd exp_data && git lfs pull
# Files end up under openpi/exp_data/pi0/*.pkl (baseline expectation)
```

> **Note:** If you have another LIBERO install, ensure `~/.libero/config.yaml` points to this repo’s `libero` and assets.

---

## 3) Runbook (two terminals)

**All paths below assume you’re developing inside `pi0-edit` but running with the baseline tree layout under `pi0-text-latent/`.**
If `pi0-edit` vendors/duplicates scripts, keep the commands identical and update paths accordingly.

### Terminal A — start the server

```bash
cd pi0-text-latent/
conda activate text-latent

# Select GPU(s) explicitly
CUDA_VISIBLE_DEVICES=4 uv run python scripts/serve_inversion_experiment.py \
  --config_name pi0_libero \
  --port 8000 \
  --verbose
```

### Terminal B — start the client

```bash
cd pi0-text-latent/
conda activate text-latent
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/modified_libero

CUDA_VISIBLE_DEVICES=4 python scripts/run_inversion_client.py \
  --port 8000 \
  --task_suite_name libero_goal \
  --task_id 9 \
  --prompt "put the wine bottle on the rack" \
  --closed_loop \
  --replan_steps 5
```

### Quick client variant (longer horizon, drawing traj, alt port)

```bash
cd pi0-text-latent/
conda activate text-latent
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/modified_libero

CUDA_VISIBLE_DEVICES=4 python scripts/run_inversion_client.py \
  --port 8001 \
  --task_suite_name libero_goal \
  --task_id 9 \
  --prompt "put the wine bottle on the rack" \
  --closed_loop \
  --max_steps 120 \
  --replan_steps 10 \
  --draw_traj
```

#### Client flag cheatsheet

* `--task_suite_name`: e.g., `libero_goal`
* `--task_id`: integer task index
* `--prompt`: natural language instruction
* `--closed_loop`: enable closed-loop control
* `--max_steps`: max env steps (default varies)
* `--replan_steps`: replan frequency
* `--draw_traj`: overlay trajectory on renders
* `--port`: must match the server port (default 8000)

> **Outputs:** By default, videos/logs align with the baseline under `pi0-text-latent/data/libero/` (unless changed in code/config).

---

## 4) Dev environment tips

* **GPU pinning:** Always set `CUDA_VISIBLE_DEVICES` on **both** server and client.
* **Two envs, two activations:**

  * Server: only conda `text-latent`.
  * Client: conda `text-latent` **then** `source examples/libero/.venv/bin/activate`.
* **PYTHONPATH:** export `third_party/modified_libero` on the **client** terminal.
* **Ports:** Default `8000`; use `8001+` for parallel runs. Ensure server and client ports match.
* **Hot-reload:** Server changes require a **server restart**. Client flag changes do **not**.

---

## 5) Testing instructions

> If/when tests are added, prefer repo-local commands. Until then:

* **Smoke test (server):**

  ```bash
  conda activate text-latent
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/serve_inversion_experiment.py --config_name pi0_libero --port 9000 --verbose
  ```

  Expect a “serving on 0.0.0.0:9000” style log.

* **Smoke test (client):**

  ```bash
  conda activate text-latent
  source examples/libero/.venv/bin/activate
  export PYTHONPATH=$PYTHONPATH:$PWD/third_party/modified_libero
  python scripts/run_inversion_client.py --port 9000 --task_suite_name libero_goal --task_id 0 --prompt "open the drawer" --closed_loop --replan_steps 3
  ```

* **CI hook (placeholder):** Add `pytest` and `ruff/black` later. For now, ensure scripts run cleanly end-to-end.

---

## 6) PR / branch conventions

* **Title:** `[pi0-edit] <short, imperative summary>`
* **Branch name:** `feat/<slug>`, `fix/<slug>`, or `exp/<slug>`
* **Before pushing:**

  * Verify **two-terminal run** works with default prompt & task.
  * Keep ports and env activations as documented.
  * Update this file if you change any run flags, ports, or entrypoints.

---

## 7) Troubleshooting

* **`ConnectionRefusedError` / client can’t reach server**

  * Port mismatch; confirm `--port` is identical on both sides.
  * Server not running or died on startup (scroll back logs).

* **`OSError: [Errno 98] Address already in use` (Linux)**

  * Kill the old process or use another port:

    ```bash
    lsof -i :8000
    kill -9 <PID>     # or switch to --port 8001
    ```

* **`ModuleNotFoundError: libero...` on client**

  * Re-activate LIBERO venv and export `PYTHONPATH`:

    ```bash
    source examples/libero/.venv/bin/activate
    export PYTHONPATH=$PYTHONPATH:$PWD/third_party/modified_libero
    ```

* **`CUDA out of memory`**

  * Try a different GPU via `CUDA_VISIBLE_DEVICES`.
  * Reduce planning load: lower `--max_steps` or `--replan_steps`.

* **LIBERO assets/dataset not found**

  * Check `~/.libero/config.yaml` points to this repo’s `libero` & assets.
  * If using text-latents from HF, ensure `git lfs pull` completed.

* **Success detection flakiness**

  * Baseline slightly relaxes contact threshold (0.03 → 0.1) and stove placement rules. Using strict upstream LIBERO can drop success (\~5%).

---

## 8) What the agent (Codex) should / should not do

* ✅ Maintain the two-terminal, two-env architecture.

* ✅ Keep CLI interfaces stable (`serve_inversion_experiment.py`, `run_inversion_client.py`).

* ✅ Add flags only if they default to backward-compatible behavior.

* ✅ Prefer explicit environment variables over implicit device selection.

* ❌ Do **not** merge the two envs or remove the LIBERO venv.

* ❌ Do **not** change default ports without updating this doc & client/server defaults.

* ❌ Do **not** break the baseline `pi0-text-latent` run paths.

---

## 9) Reference commands (copy–paste)

### Server (default)

```bash
cd pi0-text-latent/
conda activate text-latent
CUDA_VISIBLE_DEVICES=4 uv run python scripts/serve_inversion_experiment.py --config_name pi0_libero --port 8000 --verbose
```

### Client (default)

```bash
cd pi0-text-latent/
conda activate text-latent
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/modified_libero
CUDA_VISIBLE_DEVICES=4 python scripts/run_inversion_client.py --port 8000 --task_suite_name libero_goal --task_id 9 --prompt "put the wine bottle on the rack" --closed_loop --replan_steps 5
```

### Quick client

```bash
cd pi0-text-latent/
conda activate text-latent
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/modified_libero
CUDA_VISIBLE_DEVICES=4 python scripts/run_inversion_client.py --port 8001 --task_suite_name libero_goal --task_id 9 --prompt "put the wine bottle on the rack" --closed_loop --max_steps 120 --replan_steps 10 --draw_traj
```

