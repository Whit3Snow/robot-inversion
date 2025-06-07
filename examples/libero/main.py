import collections
import dataclasses
import json
import logging
import math
import os.path as osp
import pathlib
import pickle
import random
import types
from datetime import datetime

import cv2
import imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import tqdm
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

EXP_DATA_PATH = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))), "exp_data", "pi0")
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90,
    task_suite_name: str = "libero_spatial_ood"  # libero_spatial_ood, libero_object_ood, libero_goal_ood
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 10  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos or images
    save_video = True
    draw_traj = False  # Draw trajectory
    render = False

    #################################################################################################################
    # Experiments
    #################################################################################################################
    # Layer to intervene, 0 to do text embedding interpolation (TEI), "all" for interpolating hidden states (TLI),
    # Other integers for specific layer interpolation. None for turning it off.
    layer_to_intervene = "all"
    use_TEI_and_TLI = False
    mask_prompt_method = None  # options: None, "blank", "mask", "blank_with_text_latent"
    obscure_prompt = False
    obscure_prompt_layer = None
    use_only_two_prompt_for_libero_object = False
    seed: int = 7  # Random Seed (for reproducibility)


libero_object_center_prompt = "pick up the cream cheese and place it in the basket"
libero_object_top_right_prompt = "pick up the alphabet soup and place it in the basket"
task_using_center_prompt = [1, 3, 9, 2, 5]

task_hidden_states_mapping = dict(
    # libero_goal_ood
    put_the_cream_cheese_on_the_plate=("put_cheese", "plate", 24),
    put_the_cream_cheese_on_the_stove=("put_cheese", "stove", 24),
    put_the_cream_cheese_on_top_of_the_cabinet=("put_cheese", "cabinet", 24),
    put_the_wine_bottle_in_the_bowl=("put_wine", "bowl", 12),  # too close, use less steps...
    put_the_wine_bottle_on_the_plate=("put_wine", "plate", 24),
    put_the_wine_bottle_on_the_stove=("put_wine", "stove", 26),
    put_the_orange_juice_on_the_stove=("put_wine", "stove", 26),
    put_the_tomato_sauce_on_top_of_the_cabinet=("put_cheese", "cabinet", 24),
    put_the_bbq_source_on_the_plate=("put_cheese", "plate", 24),
    put_the_cream_cheese_in_the_basket=("put_cheese", "stove", 24),

    # spatial ood
    put_the_bowl_at_table_center_on_the_stove=("center_bowl", "stove", 24),
    put_the_bowl_at_table_center_on_the_cabinet=("center_bowl", "cabinet", 24),
    put_the_bowl_next_to_the_plate_on_the_stove=("next_plate_bowl", "stove", 30),
    put_the_bowl_next_to_the_plate_on_the_cabinet=("next_plate_bowl", "cabinet", 24),
    put_the_bowl_on_cookie_box_on_the_stove=("cookie_bowl", "stove", 24),
    put_the_bowl_on_cookie_box_on_the_cabinet=("cookie_bowl", "cabinet", 14),
    put_the_orange_juice_on_the_plate=("put_wine", "ramkin_bowl", 24),
    put_the_milk_on_the_plate=("put_wine", "ramkin_bowl", 24),
    put_the_chocolate_pudding_on_the_plate=("put_cheese", "cookie_bowl", 24),
    put_the_butter_on_the_plate=("put_cheese", "cookie_bowl", 24),

    # task reconstruction goal, as T-1 = T-2, so interpolation doesn't take effect
    open_the_middle_drawer_of_the_cabinet=("open_drawer", "open_drawer", 1),
    open_the_top_drawer_and_put_the_bowl_inside=("drawer_inside_bowl", "drawer_inside_bowl", 1),
    push_the_plate_to_the_front_of_the_stove=("plate_front_stove", "plate_front_stove", 1),
    put_the_cream_cheese_in_the_bowl=("cheese_in_bowl", "cheese_in_bowl", 1),
    turn_on_the_stove=("turn_on_stove", "turn_on_stove", 1),
    put_the_wine_bottle_on_the_rack=("wine_to_rack", "wine_to_rack", 1),
    put_the_bowl_on_the_stove=("put_bowl_stove", "put_bowl_stove", 1),
    put_the_bowl_on_the_plate=("put_bowl_plate", "put_bowl_plate", 1),
    put_the_wine_bottle_on_top_of_the_cabinet=("wine_to_cabinet", "wine_to_cabinet", 1),
    put_the_bowl_on_top_of_the_cabinet=("put_bowl_cabinet", "put_bowl_cabinet", 1),

    # task reconstruction object
    pick_up_the_alphabet_soup_and_place_it_in_the_basket=("alphabet_soup", "alphabet_soup", 1),
    pick_up_the_cream_cheese_and_place_it_in_the_basket=("cream_cheese", "cream_cheese", 1),
    pick_up_the_salad_dressing_and_place_it_in_the_basket=("salad_dressing", "salad_dressing", 1),
    pick_up_the_bbq_sauce_and_place_it_in_the_basket=("bba_sauce", "bba_sauce", 1),
    pick_up_the_ketchup_and_place_it_in_the_basket=("ketchup", "ketchup", 1),
    pick_up_the_tomato_sauce_and_place_it_in_the_basket=("tomato_sauce", "tomato_sauce", 1),
    pick_up_the_butter_and_place_it_in_the_basket=("butter", "butter", 1),
    pick_up_the_milk_and_place_it_in_the_basket=("milk", "milk", 1),
    pick_up_the_chocolate_pudding_and_place_it_in_the_basket=("chocalate_pudding", "chocalate_pudding", 1),
    pick_up_the_orange_juice_and_place_it_in_the_basket=("orange_juice", "orange_juice", 1),

    # task reconstruction spatial ood
    pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate=(
        "between_bowl", "between_bowl", 1),
    pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate=("next_ramkin_bowl", "next_ramkin_bowl", 1),
    pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate=("center_bowl", "center_bowl", 1),
    pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate=("cookie_bowl", "cookie_bowl", 1),
    pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate=(
        "top_drawer_bowl", "top_drawer_bowl", 1),
    pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate=("ramkin_bowl", "ramkin_bowl", 1),
    pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate=("next_cookie_bowl", "next_cookie_bowl", 1),
    pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate=("stove_bowl", "stove_bowl", 1),
    pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate=("next_plate_bowl", "next_plate_bowl", 1),
    pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate=("cabinet_bowl", "cabinet_bowl", 1),

)

hidden_states_mapping_file = dict(
    # For libero goal extrapolation
    put_wine=f"{EXP_DATA_PATH}/avg_states_14_15_frame_0_119.pkl",
    put_cheese=f"{EXP_DATA_PATH}/avg_states_13_14_frame_0_119.pkl",
    plate=f"{EXP_DATA_PATH}/avg_states_10_11_frame_0_119.pkl",
    stove=f"{EXP_DATA_PATH}/avg_states_17_18_frame_0_119.pkl",
    cabinet=f"{EXP_DATA_PATH}/avg_states_18_19_frame_0_119.pkl",
    bowl=f"{EXP_DATA_PATH}/avg_states_13_14_frame_0_119.pkl",

    # for libero-spatial task extrapolation
    center_bowl=f"{EXP_DATA_PATH}/avg_states_38_39_frame_0_119.pkl",
    between_bowl=f"{EXP_DATA_PATH}/avg_states_34_35_frame_0_119.pkl",
    next_plate_bowl=f"{EXP_DATA_PATH}/avg_states_36_37_frame_0_119.pkl",
    cookie_bowl=f"{EXP_DATA_PATH}/avg_states_35_36_frame_0_119.pkl",
    next_cookie_bowl=f"{EXP_DATA_PATH}/avg_states_30_31_frame_0_119.pkl",
    ramkin_bowl=f"{EXP_DATA_PATH}/avg_states_32_33_frame_0_119.pkl",
    next_ramkin_bowl=f"{EXP_DATA_PATH}/avg_states_37_38_frame_0_119.pkl",
    stove_bowl=f"{EXP_DATA_PATH}/avg_states_33_34_frame_0_119.pkl",
    top_drawer_bowl=f"{EXP_DATA_PATH}/avg_states_31_32_frame_0_119.pkl",
    cabinet_bowl=f"{EXP_DATA_PATH}/avg_states_39_40_frame_0_119.pkl",

    # For libero-goal task reconstruction
    put_bowl_plate=f"{EXP_DATA_PATH}/avg_states_10_11_frame_0_119.pkl",
    wine_to_rack=f"{EXP_DATA_PATH}/avg_states_11_12_frame_0_119.pkl",
    drawer_inside_bowl=f"{EXP_DATA_PATH}/avg_states_12_13_frame_0_119.pkl",
    cheese_in_bowl=f"{EXP_DATA_PATH}/avg_states_13_14_frame_0_119.pkl",
    wine_to_cabinet=f"{EXP_DATA_PATH}/avg_states_14_15_frame_0_119.pkl",
    plate_front_stove=f"{EXP_DATA_PATH}/avg_states_15_16_frame_0_119.pkl",
    turn_on_stove=f"{EXP_DATA_PATH}/avg_states_16_17_frame_0_119.pkl",
    put_bowl_stove=f"{EXP_DATA_PATH}/avg_states_17_18_frame_0_119.pkl",
    put_bowl_cabinet=f"{EXP_DATA_PATH}/avg_states_18_19_frame_0_119.pkl",
    open_drawer=f"{EXP_DATA_PATH}/avg_states_19_20_frame_0_119.pkl",

    # For libero-object task reconstruction
    orange_juice=f"{EXP_DATA_PATH}/avg_states_20_21_frame_0_119.pkl",
    ketchup=f"{EXP_DATA_PATH}/avg_states_21_22_frame_0_119.pkl",
    cream_cheese=f"{EXP_DATA_PATH}/avg_states_22_23_frame_0_119.pkl",
    bba_sauce=f"{EXP_DATA_PATH}/avg_states_23_24_frame_0_119.pkl",
    alphabet_soup=f"{EXP_DATA_PATH}/avg_states_24_25_frame_0_119.pkl",
    milk=f"{EXP_DATA_PATH}/avg_states_25_26_frame_0_119.pkl",
    salad_dressing=f"{EXP_DATA_PATH}/avg_states_26_27_frame_0_119.pkl",
    butter=f"{EXP_DATA_PATH}/avg_states_27_28_frame_0_119.pkl",
    tomato_sauce=f"{EXP_DATA_PATH}/avg_states_28_29_frame_0_119.pkl",
    chocalate_pudding=f"{EXP_DATA_PATH}/avg_states_29_30_frame_0_119.pkl",
)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
marker_params = []


def interpolate_colors(color1, color2, num_points=100):
    color1 = np.array(color1)
    color2 = np.array(color2)

    return [tuple(color1 + (color2 - color1) * t)
            for t in np.linspace(0, 1, num_points)]


red_color = [209 / 255, 32 / 255, 32 / 255, 1]
blue_color = [0, 122 / 255, 206 / 255, 255 / 255]
green_color = [38 / 255, 138 / 255, 33 / 255, 1]
orange_color = [213 / 255, 148 / 255, 0, 1]

num_points = 100
# Create a colormap object
cmap = plt.get_cmap('viridis')
# points = [0.9]
# points = [0.1]
points = np.linspace(0, 1, 100)  # even sample
# points = np.random.normal(loc=0.45, scale=0.15, size=100)
# points = np.clip(points, 0, 1)
# points = np.sort(np.random.choice(points, num_points, replace=False))
color_list = [cmap(p) for p in points]


# color_list = [blue_color]
# color_list = interpolate_colors(green_color, blue_color, num_points)


def add_marker_to_scene(render_context, marker_params):
    """ Adds marker to scene, and returns the corresponding object. """
    if render_context.scn.ngeom >= render_context.scn.maxgeom:
        raise RuntimeError('Ran out of geoms. maxgeom: %d' % render_context.scn.maxgeom)

    for param in marker_params:
        g = render_context.scn.geoms[render_context.scn.ngeom]
        g.dataid = -1
        g.objtype = 0  # mujoco.OBJ_UNKNOWN
        g.objid = -1
        g.category = 4  # const.CAT_DECOR
        g.emission = 0.5
        g.specular = 0
        g.shininess = 0.
        g.transparent = 0
        g.reflectance = 0
        g.label = ""
        g.type = 2  # const.GEOM_SPHERE
        g.size[:] = np.array([0.008, 0.008, 0.008])
        g.mat[:] = np.eye(3)
        g.matid = -1
        g.pos[:] = param["pos"]
        g.rgba[:] = param["rgba"]

        render_context.scn.ngeom += 1


def render(self, width, height, camera_id=None, segmentation=False):
    viewport = mujoco.MjrRect(0, 0, width, height)

    # if self.sim.render_callback is not None:
    #     self.sim.render_callback(self.sim, self)

    # update width and height of rendering context if necessary
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
        self.model._model, self.data._data, self.vopt, self.pert, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn
    )

    if segmentation:
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 1
        self.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1

    add_marker_to_scene(self, marker_params)

    mujoco.mjr_render(viewport=viewport, scn=self.scn, con=self.con)
    # for gridpos, (text1, text2) in self._overlay.items():
    #     mjr_overlay(const.FONTSCALE_150, gridpos, rect, text1.encode(), text2.encode(), &self._con)

    if segmentation:
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 0
        self.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 0


def eval_libero(args: Args) -> None:
    if args.use_only_two_prompt_for_libero_object:
        assert args.task_suite_name == "libero_object"
        assert args.layer_to_intervene is None
        assert args.mask_prompt_method is None
        assert not args.obscure_prompt

    if args.obscure_prompt:
        assert args.mask_prompt_method is None
        assert args.layer_to_intervene is None
        with open(f"{EXP_DATA_PATH}/reconstruct_prompt.pkl", "rb") as f:
            reconstruct_prompt = pickle.load(f)

    if args.mask_prompt_method is not None:
        assert not args.obscure_prompt

    if args.draw_traj:
        assert not args.save_video and args.render
        LIBERO_ENV_RESOLUTION = 1024  # for making demo
    else:
        LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

    # Set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    # append current time to the video path with lib date
    args.video_out_path = args.video_out_path.replace("videos", args.task_suite_name)
    args.video_out_path += datetime.now().strftime("-%m-%d-%H-%M")
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial" or args.task_suite_name == "libero_spatial_ood":
        max_steps = 300  # longest training demo has 193 steps, NOTE: ood has longer movement
    elif args.task_suite_name == "libero_object" or args.task_suite_name == "libero_object_ood":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal" or args.task_suite_name == "libero_goal_ood":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    results = {}
    total_episodes, total_successes, total_steps = 0, 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        # initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        if args.save_video:
            task_segment = task_description.replace(" ", "_")
            (pathlib.Path(args.video_out_path) / f"{task_segment}").mkdir(parents=True, exist_ok=False)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            obs = env.reset()
            env.env.sim._render_context_offscreen.render = types.MethodType(render,
                                                                            env.env.sim._render_context_offscreen)
            marker_params.clear()
            reset_sever = True

            action_plan = collections.deque()

            # Set initial states
            # obs = env.set_init_state(initial_states[episode_idx])

            print(str(task_description))
            # Setup
            t = 0
            replay_images = []
            replay_wrist_images = []

            logging.info(f"Starting episode {task_episodes + 1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        # cv2.imwrite("test.png", cv2.cvtColor(obs["agentview_image"][::-1, ::-1], cv2.COLOR_BGR2RGB))
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    original_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    original_wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(original_img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(original_wrist_img, args.resize_size, args.resize_size)
                    )
                    if args.render:
                        cv2.imshow("agentview", cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
                        cv2.imshow("robot0_eye_in_hand", cv2.cvtColor(original_wrist_img, cv2.COLOR_BGR2RGB))
                        cv2.waitKey(1)
                    # Save preprocessed image for replay video
                    replay_images.append(original_img)
                    replay_wrist_images.append(original_wrist_img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "done": reset_sever,
                            "layer_to_intervene": args.layer_to_intervene,
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                            "mask_prompt_method": args.mask_prompt_method,
                            "use_TEI_and_TLI": args.use_TEI_and_TLI,
                            "task_hidden_states_mapping": task_hidden_states_mapping,
                            "hidden_states_mapping_file": hidden_states_mapping_file,
                        }
                        if args.use_only_two_prompt_for_libero_object:
                            if task_id in task_using_center_prompt:
                                element["prompt"] = libero_object_center_prompt
                            else:
                                element["prompt"] = libero_object_top_right_prompt
                            if reset_sever:
                                print(f"replace object prompt with {element['prompt']}")

                        if args.obscure_prompt:
                            if args.obscure_prompt_layer is None:
                                if args.task_suite_name == "libero_object":
                                    layer = 2
                                elif args.task_suite_name == "libero_goal":
                                    layer = 1
                                elif args.task_suite_name == "libero_spatial":
                                    layer = 3
                                else:
                                    raise ValueError(f"Unknown task suite: {args.task_suite_name}")
                            else:
                                layer = args.obscure_prompt_layer
                            element["prompt_to_use"] = reconstruct_prompt[str(task_segment)]["token_indices"][layer]
                            element["prompt_to_use"][0] = 2  # match the distribution
                            element["prompt_to_use"][-1] = 108  # match the distribution
                            element["prompt_to_use_str"] = reconstruct_prompt[str(task_segment)]["decoding_result"][
                                layer]

                        # Query model to get action
                        received = client.infer(element)
                        action_chunk = received["actions"]
                        reset_sever = False
                        assert (
                                len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                        total_steps += 1

                    action = action_plan.popleft()

                    if args.draw_traj and t % 2 == 0:
                        color = color_list[t] if t < len(color_list) else color_list[-1]
                        to_draw_data = dict(pos=obs["robot0_eef_pos"], rgba=np.array(color))
                        marker_params.append(to_draw_data)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")

            if args.save_video:
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path) / f"{task_segment}/{suffix}_{episode_idx}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=30,  # the same as openvla
                )
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path) / f"{task_segment}/{suffix}_wrist_{episode_idx}.mp4",
                    [np.asarray(x) for x in replay_wrist_images],
                    fps=30)  # the same as openvla

            if args.draw_traj:
                dir = pathlib.Path(args.video_out_path) / f"{task_segment}/{suffix}_{episode_idx}"
                dir.mkdir(parents=True, exist_ok=True)
                for idx, img in enumerate(replay_images):
                    cv2.imwrite(str(dir / f"{idx}.png"), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"Intervention: {args.layer_to_intervene}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            logging.info(f"mean steps so far: {total_steps / total_episodes:.1f}")

        # Log final results
        task_segment = task_description.replace(" ", "_")
        results[task_segment] = task_successes / task_episodes
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    results["total_success_rate"] = float(total_successes) / float(total_episodes)
    results["total_episodes"] = total_episodes
    if args.obscure_prompt:
        results["reconstruct_prompt"] = reconstruct_prompt
    results["args"] = dataclasses.asdict(args)

    to_intervene = args.layer_to_intervene or "none"
    with open(pathlib.Path(args.video_out_path) / f"result_{to_intervene}.json", "w") as f:
        json.dump(results, f, indent=4)


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def run_reconstruction_exp():
    logging.basicConfig(level=logging.INFO)
    for task_suite in ["libero_goal", "libero_spatial", "libero_object"]:
        # Text latent + blank prompt experiments
        args = Args()
        args.task_suite_name = task_suite
        args.mask_prompt_method = "blank_with_text_latent"
        args.layer_to_intervene = None
        eval_libero(args)

        # obscure prompt experiments
        for layer in [1, 2, 3] if task_suite != "libero_goal" else [1, 2]:  # 3 would fail drastically, no need to run
            args = Args()
            args.task_suite_name = task_suite
            args.layer_to_intervene = None
            args.obscure_prompt = True
            args.obscure_prompt_layer = layer
            eval_libero(args)

        # Baseline 1
        args = Args()
        args.task_suite_name = task_suite
        args.mask_prompt_method = "blank"
        args.layer_to_intervene = None
        eval_libero(args)

        # Baseline 2
        args = Args()
        args.task_suite_name = task_suite
        args.mask_prompt_method = "mask"
        args.layer_to_intervene = None
        eval_libero(args)


def run_extrapolation_exp():
    logging.basicConfig(level=logging.INFO)
    for task_suite in ["libero_spatial_ood", "libero_goal_ood"]:
        # TLI
        args = Args()
        args.task_suite_name = task_suite
        args.layer_to_intervene = "all"
        eval_libero(args)

        # TEI
        args = Args()
        args.task_suite_name = task_suite
        args.layer_to_intervene = 0
        eval_libero(args)

        # TEI + TLI
        args = Args()
        args.task_suite_name = task_suite
        args.layer_to_intervene = "all"
        args.use_TEI_and_TLI = True
        eval_libero(args)

        # TLI + blank prompt
        args = Args()
        args.task_suite_name = task_suite
        args.layer_to_intervene = "all"
        args.mask_prompt_method = "blank"
        eval_libero(args)


if __name__ == "__main__":
    # run_reconstruction_exp()
    run_extrapolation_exp()
