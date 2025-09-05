import dataclasses
import pathlib
from datetime import datetime
import pickle
import typing as tp
from typing import Optional, Tuple, List
import math
import logging

import numpy as np
import tyro

import os
import sys
import pathlib as _p

from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

try:
    import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # fallback: no progress bars


# ------------------------------
# Visualization helpers (shared)
# ------------------------------
def _compute_color_list(num_steps: int) -> List[tp.Tuple[float, float, float, float]]:
    if num_steps <= 0:
        return []
    try:
        import matplotlib.pyplot as plt  # type: ignore

        cmap = plt.get_cmap("viridis")
        points = np.linspace(0, 1, num_steps)
        return [cmap(p) for p in points]
    except Exception:
        return [(0.2 + 0.8 * i / max(1, num_steps), 0.1, 0.9 - 0.8 * i / max(1, num_steps), 1.0) for i in range(num_steps)]


def _add_marker_to_scene(render_context, marker_params: List[dict]) -> None:
    if render_context.scn.ngeom >= render_context.scn.maxgeom:
        return
    for param in marker_params:
        g = render_context.scn.geoms[render_context.scn.ngeom]
        g.dataid = -1
        g.objtype = 0
        g.objid = -1
        g.category = 4
        g.emission = 0.5
        g.specular = 0
        g.shininess = 0.0
        g.transparent = 0
        g.reflectance = 0
        g.label = ""
        g.type = 2
        g.size[:] = np.array([0.008, 0.008, 0.008])
        g.mat[:] = np.eye(3)
        g.matid = -1
        g.pos[:] = param["pos"]
        g.rgba[:] = param["rgba"]
        render_context.scn.ngeom += 1


def _get_render_with_markers(marker_params: List[dict]):
    import mujoco  # lazy import

    def render(self, width, height, camera_id=None, segmentation=False):
        viewport = mujoco.MjrRect(0, 0, width, height)
        if width > self.con.offWidth or height > self.con.offHeight:
            new_width = max(width, self.model.vis.global_.offwidth)
            new_height = max(height, self.model.vis.global_.offheight)
            self.update_offscreen_size(new_width, new_height)
        if camera_id is not None:
            if camera_id == -1:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = camera_id
        mujoco.mjv_updateScene(
            self.model._model,
            self.data._data,
            self.vopt,
            self.pert,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scn,
        )
        _add_marker_to_scene(self, marker_params)
        mujoco.mjr_render(viewport=viewport, scn=self.scn, con=self.con)

    return render


def _patch_offscreen_render_with_markers(env, marker_params: List[dict]) -> None:
    import types as _types

    render = _get_render_with_markers(marker_params)
    env.env.sim._render_context_offscreen.render = _types.MethodType(
        render, env.env.sim._render_context_offscreen
    )


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8001

    task_suite_name: str = "libero_goal"
    task_id: int = 0
    prompt: Optional[str] = None
    resize_size: int = 224
    seed: int = 42

    save_results: bool = True
    # Base directory for experiment outputs
    base_out_dir: str = "/home/nas4_user/hyojinjang/repos/sangmin/pi0-text-latent/data/libero"
    # Optional experiment name; if empty, will be generated from suite
    exp_name: Optional[str] = None
    save_video: bool = True
    closed_loop: bool = False
    replan_steps: int = 5
    max_steps: Optional[int] = None
    draw_traj: bool = False



def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = float(np.sqrt(1.0 - quat[3] * quat[3]))
    if np.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * float(np.arccos(quat[3]))) / den


def get_libero_obs_dict(
    *, task_suite_name: str, task_id: int, prompt: Optional[str], resize_size: int, seed: int
) -> dict:
    """Create observation dictionary from Libero environment."""
    print(f"🔧 Initializing Libero environment...")
    print(f"   Task suite: {task_suite_name}, Task ID: {task_id}")
    
    # Lazy-import Libero and add repo path so the client script imports work only when needed.
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), "..", "third_party", "modified_libero"))
        from libero.libero import benchmark  # type: ignore
        from libero.libero import get_libero_path  # type: ignore
        from libero.libero.envs import OffScreenRenderEnv  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Libero/robosuite not found in the client environment.\n"
            "Please install client deps in a separate venv (Python 3.8 recommended):\n"
            "  uv venv --python 3.8 examples/libero/.venv\n"
            "  source examples/libero/.venv/bin/activate\n"
            "  uv pip sync examples/libero/requirements.txt third_party/modified_libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match\n"
            "  uv pip install -e packages/openpi-client\n"
            "  uv pip install -e third_party/modified_libero\n"
            "  export PYTHONPATH=$PYTHONPATH:$PWD/third_party/modified_libero\n"
        ) from e

    print(f"   Loading benchmark and task...")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    task_description = task.language
    if prompt is None:
        prompt = task_description
    
    print(f"   Task description: '{task_description}'")
    print(f"   Using prompt: '{prompt}'")

    print(f"   Creating environment...")
    task_bddl_file = _p.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 256,
        "camera_widths": 256,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)

    print(f"   Resetting environment and stabilizing...")
    obs = env.reset()
    # Stabilize
    LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
    for i in range(10):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        if i % 3 == 0:
            print(f"      Stabilization step {i+1}/10...")

    print(f"   Processing images and state...")
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize_size, resize_size))

    state = np.concatenate(
        [obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]
    ).astype(np.float32)

    env.close()
    print(f"   ✅ Observation ready! State shape: {state.shape}, Image shape: {img.shape}")

    return {
        "observation/state": state,
        "observation/image": img,
        "observation/wrist_image": wrist,
        "prompt": str(prompt),
    }


def _obs_dict_from_env_obs(obs: dict, prompt: str, resize_size: int) -> dict:
    """Builds an observation dict expected by the inversion server from a live env obs."""
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize_size, resize_size))
    state = np.concatenate(
        [obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]
    ).astype(np.float32)
    return {
        "observation/state": state,
        "observation/image": img,
        "observation/wrist_image": wrist,
        "prompt": str(prompt),
    }


def _rollout_and_record(
    *,
    task_suite_name: str,
    task_id: int,
    prompt: str,
    resize_size: int,
    seed: int,
    actions_7d: np.ndarray,
    max_steps: Optional[int] = None,
    draw_traj: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Roll out a single episode with given 7D actions and return list of frames (agentview)."""
    print(f"   🎬 Setting up rollout environment...")
    
    # Lazy import libero and set path
    import types
    import cv2
    import numpy as np
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "third_party", "modified_libero"))
    from libero.libero import benchmark  # type: ignore
    from libero.libero import get_libero_path  # type: ignore
    from libero.libero.envs import OffScreenRenderEnv  # type: ignore
    import mujoco

    # Color palette and marker params
    marker_params: List[dict] = []

    # Build task and env
    print(f"      Creating task environment...")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    task_bddl_file = _p.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=task_bddl_file, camera_heights=256, camera_widths=256)
    env.seed(seed)

    print(f"      Resetting and stabilizing...")
    obs = env.reset()
    if draw_traj:
        _patch_offscreen_render_with_markers(env, marker_params)

    # Stabilize
    LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
    for _ in range(10):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

    frames_agent: List[np.ndarray] = []
    frames_wrist: List[np.ndarray] = []
    steps = len(actions_7d) if max_steps is None else min(max_steps, len(actions_7d))
    color_list = _compute_color_list(steps) if draw_traj else []
    
    print(f"      Rolling out {steps} action steps...")
    for t in range(steps):
        if t % 10 == 0 or t == steps - 1:
            print(f"         Step {t+1}/{steps}")
            
        # Record
        original_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        frames_agent.append(original_img)
        frames_wrist.append(wrist_img)

        # Add trajectory markers if enabled
        if draw_traj and t < len(color_list):
            color = color_list[t]
            marker_params.append(dict(pos=obs["robot0_eef_pos"], rgba=np.array(color)))

        # Step
        action = actions_7d[t].tolist()
        obs, _, done, _ = env.step(action)
        if done:
            print(f"         Task completed at step {t+1}!")
            break

    env.close()
    print(f"      ✅ Rollout complete! Recorded {len(frames_agent)} frames")
    return frames_agent, frames_wrist


def _default_max_steps_for_suite(task_suite_name: str) -> int:
    if task_suite_name in ("libero_spatial", "libero_spatial_ood"):
        return 300
    if task_suite_name in ("libero_object", "libero_object_ood"):
        return 280
    if task_suite_name in ("libero_goal", "libero_goal_ood"):
        return 300
    if task_suite_name == "libero_10":
        return 520
    if task_suite_name == "libero_90":
        return 400
    return 300


def run_closed_loop_episode(
    *, host: str, port: int, task_suite_name: str, task_id: int, prompt: str, resize_size: int, seed: int,
    replan_steps: int, max_steps: Optional[int], use_reconstructed_for_control: bool, draw_traj: bool = False,
    desc: str = "episode",
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], bool]:
    """Closed-loop rollout with periodic replanning.

    Per chunk, compute MSE over the executed segment (first replan_steps) between
    the server-returned plans: result["actions"] vs result["reconstructed_actions"].
    """
    import types
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "third_party", "modified_libero"))
    from libero.libero import benchmark  # type: ignore
    from libero.libero import get_libero_path  # type: ignore
    from libero.libero.envs import OffScreenRenderEnv  # type: ignore

    client = _websocket_client_policy.WebsocketClientPolicy(host=host, port=port)

    # Build env
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    task_bddl_file = _p.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=task_bddl_file, camera_heights=256, camera_widths=256)
    env.seed(seed)

    obs = env.reset()
    # Stabilize
    LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
    for _ in range(10):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

    frames_agent: List[np.ndarray] = []
    frames_wrist: List[np.ndarray] = []
    mse_list: List[float] = []
    done = False
    t = 0
    max_steps = max_steps or _default_max_steps_for_suite(task_suite_name)
    
    # Setup trajectory visualization (color palette + render patch) if enabled
    marker_params: List[dict] = []
    if draw_traj:
        color_list = _compute_color_list(max_steps)
        _patch_offscreen_render_with_markers(env, marker_params)
    else:
        color_list = []
    total_replans = math.ceil(max_steps / max(1, replan_steps))

    # progress bars (if tqdm is available)
    pbar_steps = tqdm.tqdm(total=max_steps, desc=f"{desc} steps") if tqdm else None
    pbar_replans = tqdm.tqdm(total=total_replans, desc=f"{desc} replans") if tqdm else None

    action_queue: List[List[float]] = []
    while t < max_steps and not done:
        # Replan if needed
        if not action_queue:
            obs_dict = _obs_dict_from_env_obs(obs, prompt, resize_size)
            result = client.infer(obs_dict)
            # Plans for this chunk
            plan_orig = np.asarray(result["actions"])           # [H,7]
            plan_recon = np.asarray(result["reconstructed_actions"])  # [H,7]
            # Execute only the first n steps from the chosen plan
            n = min(replan_steps, len(plan_orig), len(plan_recon))
            chosen = plan_recon if use_reconstructed_for_control else plan_orig
            action_queue = chosen[:n].tolist()
            # Per-chunk MSE on executed segment
            mse_chunk = float(np.mean((plan_orig[:n] - plan_recon[:n]) ** 2))
            mse_list.append(mse_chunk)
            if pbar_replans:
                pbar_replans.update(1)

        # Record current frame
        agent_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        frames_agent.append(agent_img)
        frames_wrist.append(wrist_img)

        # Add a trajectory marker for the executed step (only valid executed path)
        if draw_traj:
            color = color_list[min(t, len(color_list) - 1)] if len(color_list) > 0 else (0.0, 1.0, 0.0, 1.0)
            marker_params.append(dict(pos=obs["robot0_eef_pos"], rgba=np.array(color)))

        # Step one action
        action = action_queue.pop(0)
        obs, _, done, _ = env.step(action)
        t += 1
        if pbar_steps:
            pbar_steps.update(1)

    env.close()
    if pbar_steps:
        pbar_steps.close()
    if pbar_replans:
        pbar_replans.close()
    return frames_agent, frames_wrist, mse_list, done


def main(args: Args) -> None:
    print("="*60)
    print("🚀 Pi0 Action Inversion & Reconstruction Experiment")
    print("="*60)
    print(f"📡 Connecting to server at {args.host}:{args.port}")
    
    client = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    
    print(f"🔍 Creating observation from Libero environment...")
    obs_dict = get_libero_obs_dict(
        task_suite_name=args.task_suite_name,
        task_id=args.task_id,
        prompt=args.prompt,
        resize_size=args.resize_size,
        seed=args.seed,
    )

    print(f"📤 Sending observation to server for inversion experiment...")
    result = client.infer(obs_dict)

    print("\n" + "="*60)
    print("📊 EXPERIMENT RESULTS RECEIVED")
    print("="*60)
    print(f"✨ Reconstruction loss: {result['reconstruction_loss']:.6f}")
    print(f"📏 Action horizon: {result['action_horizon']}, Action dim: {result['action_dim']}")
    print(f"🎯 Actions preview (first 3 steps):")
    print(f"{np.asarray(result['actions'])[:3]}")

    # Build experiment directory path
    ts = datetime.now().strftime("-%m-%d-%H-%M")
    default_name = f"pi0_inv_recon_{args.task_suite_name}"
    exp_name = (args.exp_name or default_name) + ts
    exp_dir = pathlib.Path(args.base_out_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Experiment directory: {exp_dir}")

    if args.save_results:
        print(f"💾 Saving results...")
        out_file = exp_dir / f"result_task{args.task_id}.pkl"
        with open(out_file, "wb") as f:
            pickle.dump(result, f)
        print(f"   ✅ Results saved to: {out_file}")

    if args.save_video and not args.closed_loop:
        print(f"\n🎥 Creating comparison videos...")
        import imageio
        # Use the properly formatted actions (should already be correct dimension)
        orig_actions = np.asarray(result["original_actions"])
        recon_actions = np.asarray(result["reconstructed_actions"])
        
        print(f"   ✅ Original actions shape: {orig_actions.shape}")
        print(f"   ✅ Reconstructed actions shape: {recon_actions.shape}")
        print(f"   🔍 Action dimension from server: {result['action_dim']}")
        
        # Verify we have the right dimensions for Libero (should be 7D)
        if orig_actions.shape[1] != 7:
            print(f"   ⚠️  WARNING: Expected 7D actions for Libero, got {orig_actions.shape[1]}D")

        # Rollout both from the same initial condition
        print("🎬 Rolling out original action plan...")
        frames_orig_agent, frames_orig_wrist = _rollout_and_record(
            task_suite_name=args.task_suite_name,
            task_id=args.task_id,
            prompt=args.prompt or "",
            resize_size=args.resize_size,
            seed=args.seed,
            actions_7d=orig_actions,
            draw_traj=args.draw_traj,
        )
        print("🎬 Rolling out reconstructed action plan...")
        frames_recon_agent, frames_recon_wrist = _rollout_and_record(
            task_suite_name=args.task_suite_name,
            task_id=args.task_id,
            prompt=args.prompt or "",
            resize_size=args.resize_size,
            seed=args.seed,
            actions_7d=recon_actions,
            draw_traj=args.draw_traj,
        )

        print(f"🎞️  Creating side-by-side comparison videos...")
        # Make side-by-side frames
        side_by_side = []
        L = min(len(frames_orig_agent), len(frames_recon_agent))
        print(f"   Processing {L} frames for agent view...")
        for i in range(L):
            if i % 10 == 0:
                print(f"      Frame {i+1}/{L}")
            left = frames_orig_agent[i]
            right = frames_recon_agent[i]
            h = max(left.shape[0], right.shape[0])
            w = left.shape[1] + right.shape[1]
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            canvas[: left.shape[0], : left.shape[1]] = left
            canvas[: right.shape[0], left.shape[1] : left.shape[1] + right.shape[1]] = right
            side_by_side.append(canvas)

        out_path = exp_dir / f"compare_task{args.task_id}.mp4"
        imageio.mimwrite(out_path, [np.asarray(x) for x in side_by_side], fps=30)
        print(f"   ✅ Agent view comparison saved to: {out_path}")

        # Wrist side-by-side
        side_by_side_wrist = []
        Lw = min(len(frames_orig_wrist), len(frames_recon_wrist))
        print(f"   Processing {Lw} frames for wrist view...")
        for i in range(Lw):
            if i % 10 == 0:
                print(f"      Frame {i+1}/{Lw}")
            left = frames_orig_wrist[i]
            right = frames_recon_wrist[i]
            h = max(left.shape[0], right.shape[0])
            w = left.shape[1] + right.shape[1]
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            canvas[: left.shape[0], : left.shape[1]] = left
            canvas[: right.shape[0], left.shape[1] : left.shape[1] + right.shape[1]] = right
            side_by_side_wrist.append(canvas)

        out_path_w = exp_dir / f"compare_wrist_task{args.task_id}.mp4"
        imageio.mimwrite(out_path_w, [np.asarray(x) for x in side_by_side_wrist], fps=30)
        print(f"   ✅ Wrist view comparison saved to: {out_path_w}")

    if args.closed_loop:
        print("\n🔁 Running closed-loop episodes (original vs reconstructed) with replanning...")
        # Episode with original actions per chunk
        frames_agent_orig, frames_wrist_orig, mse_list_orig, success_orig = run_closed_loop_episode(
            host=args.host,
            port=args.port,
            task_suite_name=args.task_suite_name,
            task_id=args.task_id,
            prompt=args.prompt or "",
            resize_size=args.resize_size,
            seed=args.seed,
            replan_steps=args.replan_steps,
            max_steps=args.max_steps,
            use_reconstructed_for_control=False,
            draw_traj=args.draw_traj,
            desc="orig",
        )
        # Episode with reconstructed actions per chunk
        frames_agent_recon, frames_wrist_recon, mse_list_recon, success_recon = run_closed_loop_episode(
            host=args.host,
            port=args.port,
            task_suite_name=args.task_suite_name,
            task_id=args.task_id,
            prompt=args.prompt or "",
            resize_size=args.resize_size,
            seed=args.seed,
            replan_steps=args.replan_steps,
            max_steps=args.max_steps,
            use_reconstructed_for_control=True,
            draw_traj=args.draw_traj,
            desc="recon",
        )

        # Save side-by-side comparison videos
        import imageio
        # Agent view
        Lc = min(len(frames_agent_orig), len(frames_agent_recon))
        side_by_side_closed = []
        for i in range(Lc):
            left = frames_agent_orig[i]
            right = frames_agent_recon[i]
            h = max(left.shape[0], right.shape[0])
            w = left.shape[1] + right.shape[1]
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            canvas[: left.shape[0], : left.shape[1]] = left
            canvas[: right.shape[0], left.shape[1] : left.shape[1] + right.shape[1]] = right
            side_by_side_closed.append(canvas)
        out_path_cl = exp_dir / f"closed_loop_compare_agent_task{args.task_id}.mp4"
        imageio.mimwrite(out_path_cl, [np.asarray(x) for x in side_by_side_closed], fps=30)
        print(f"   ✅ Closed-loop agent comparison saved to: {out_path_cl}")

        # Wrist view
        Lcw = min(len(frames_wrist_orig), len(frames_wrist_recon))
        side_by_side_closed_w = []
        for i in range(Lcw):
            left = frames_wrist_orig[i]
            right = frames_wrist_recon[i]
            h = max(left.shape[0], right.shape[0])
            w = left.shape[1] + right.shape[1]
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            canvas[: left.shape[0], : left.shape[1]] = left
            canvas[: right.shape[0], left.shape[1] : left.shape[1] + right.shape[1]] = right
            side_by_side_closed_w.append(canvas)
        out_path_clw = exp_dir / f"closed_loop_compare_wrist_task{args.task_id}.mp4"
        imageio.mimwrite(out_path_clw, [np.asarray(x) for x in side_by_side_closed_w], fps=30)
        print(f"   ✅ Closed-loop wrist comparison saved to: {out_path_clw}")

        # Save metrics for both runs
        metrics_path = exp_dir / f"closed_loop_metrics_task{args.task_id}.pkl"
        with open(metrics_path, "wb") as f:
            pickle.dump({
                "mse_per_chunk_original": mse_list_orig,
                "mse_per_chunk_reconstructed": mse_list_recon,
                "success_original": success_orig,
                "success_reconstructed": success_recon,
                "replan_steps": args.replan_steps,
            }, f)
        print(f"   ✅ Closed-loop metrics saved to: {metrics_path}")

    print("\n" + "="*60)
    print("🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main(tyro.cli(Args))
