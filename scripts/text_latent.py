import dataclasses
import os.path
import pickle

import jax
import jax.experimental
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import tqdm
from toolz.tests.test_dicttoolz import defaultdict

import openpi.models.model as _model
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.sharding as sharding
import openpi.transforms as _transforms
from openpi import EXP_DATA_PATH
from openpi.models.pi0 import Pi0
from openpi.shared import nnx_utils
from openpi.shared.download import maybe_download
from openpi.training.data_loader import TorchDataLoader, TransformedDataset


def normalize(vectors):
    return vectors / jnp.linalg.norm(vectors, axis=1, keepdims=True)


@dataclasses.dataclass
class Checkpoint:
    config: str
    dir: str


@dataclasses.dataclass
class Args:
    default_prompt: str | None = None
    policy = None


def create_dataloader(train_config, data_config, ckpt_dir, task_range, batch_size, episode_to_use_per_task, debug):
    norm_stats = _checkpoints.load_norm_stats(ckpt_dir / "assets", data_config.asset_id)
    mesh = sharding.make_mesh(train_config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id, local_files_only=False)
    episode_per_task = defaultdict(list)
    for index, episode in enumerate(dataset_meta.episodes):
        task_desc = episode["tasks"][0]
        if task_range[0] <= dataset_meta.task_to_task_index[task_desc] < task_range[1]:
            episode_per_task[task_desc].append(index)

    assert len(episode_per_task) == task_range[1] - task_range[0], "episode picking error"
    episode_to_use = []
    for key in episode_per_task:
        episode_to_use += episode_per_task[key][:episode_to_use_per_task] \
            if episode_to_use_per_task else episode_per_task[key]

    delta_time = {key: [t / dataset_meta.fps for t in range(train_config.model.action_horizon)]
                  for key in data_config.action_sequence_keys}
    dataset = lerobot_dataset.LeRobotDataset(data_config.repo_id, delta_timestamps=delta_time,
                                             local_files_only=data_config.local_files_only,
                                             episodes=episode_to_use)
    assert data_config.norm_stats is None, "we will overwrite the norm stats with the training one"
    dataset = TransformedDataset(dataset, [
        _transforms.PromptFromLeRobotTask(dataset_meta.tasks),
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs])
    local_batch_size = batch_size
    data_loader = TorchDataLoader(dataset,
                                  local_batch_size=local_batch_size,
                                  sharding=data_sharding,
                                  shuffle=False,
                                  num_workers=0 if debug else 8,
                                  drop_last=False,
                                  seed=train_config.seed)

    # create policy
    num_iters = len(data_loader.torch_loader)
    data_loader.set_num_batches(num_iters)
    return data_loader, dataset_meta


if __name__ == '__main__':
    task_ids_to_collect = (10, 40)
    episode_to_use_for_collection = 20
    num_reconstruct_layer = 4
    reconstruct_ret_dir = "{}/pi0/".format(EXP_DATA_PATH)

    assert not os.path.exists(os.path.join(EXP_DATA_PATH, "pi0")), \
        f"Directory {EXP_DATA_PATH}/pi0 already exists. Please remove it before running this script again."

    # load policy
    prefix_len = 816  # 256 * 3 (image) + 48 (text)
    args = Args()
    args.policy = Checkpoint(config=f"pi0_libero", dir="s3://openpi-assets/checkpoints/pi0_libero")
    train_config = _config.get_config(args.policy.config)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    ckpt_dir = maybe_download(args.policy.dir)
    model: Pi0 = train_config.model.load(_model.restore_params(ckpt_dir / "params", dtype=jnp.bfloat16))
    compute_loss = nnx_utils.module_jit(model.compute_loss_with_extra)

    # config for data collection
    for task_id in range(*task_ids_to_collect):
        print(f"Generate task tensor for task_id: {task_id}")
        task_range = (task_id, task_id + 1)
        frame_index_to_use = [i for i in range(120)]  # it is wide enough to cover all frames
        data_loader, dataset_meta = create_dataloader(train_config, data_config, ckpt_dir, task_range, 1,
                                                      episode_to_use_for_collection, True)
        dataset = data_loader.torch_loader.dataset
        data_loader.set_num_batches(len(data_loader.torch_loader))


        def decode(obs):
            if not isinstance(obs, list):
                obs = obs.tokenized_prompt[0].tolist()
            return dataset._transform.transforms[-1].tokenizer._tokenizer.decode(obs)


        steps = 0
        hidden_states_sum = jnp.zeros((18, prefix_len, 2048), dtype=jnp.float32)
        post_attn_sum = jnp.zeros((18, prefix_len, 2048), dtype=jnp.float32)
        post_attn_embedding_sum = jnp.zeros((18, prefix_len, 2048), dtype=jnp.float32)
        text_token_head_output_sum = jnp.zeros((18, 48, 8, 2048), dtype=jnp.float32)
        for batch in tqdm.tqdm(data_loader, total=len(data_loader.torch_loader), desc=f"{steps}"):
            if frame_index_to_use is not None and batch["frame_index"][0] not in frame_index_to_use:
                continue
            # padded action may happen, but the training code didn't do masking
            actions = batch.pop("actions")
            obs = _model.Observation.from_dict(batch)
            result = compute_loss(jax.random.key(train_config.seed), obs, actions)
            hidden_states_sum += result["hidden_states_sum"]
            post_attn_sum += result["post_attn_sum"]
            post_attn_embedding_sum += result["post_attn_embedding_sum"]
            text_token_head_output_sum += result["text_representation"]
            steps += actions.shape[0]

        if frame_index_to_use is None:
            assert steps == len(data_loader.torch_loader.dataset), "steps not equal to dataset length"
        print(decode(obs.tokenized_prompt[0].tolist()))
        hidden_states_avg = hidden_states_sum.at[...].divide(steps)
        post_attn_avg = post_attn_sum.at[...].divide(steps)
        post_attn_embedding_avg = post_attn_embedding_sum.at[...].divide(steps)
        text_token_head_output_avg = text_token_head_output_sum.at[...].divide(steps)

        save_dir = f"{EXP_DATA_PATH}/pi0"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        start = frame_index_to_use[0] if frame_index_to_use is not None else 0
        end = frame_index_to_use[-1] if frame_index_to_use is not None else "all"
        file_id = "{}/pi0/avg_states_{}_{}_frame_{}_{}.pkl".format(EXP_DATA_PATH, *task_range, start, end)
        with open(file_id, "wb") as f:
            pickle.dump(dict(hidden_states_avg=hidden_states_avg,
                             post_attn_avg=post_attn_avg,
                             post_attn_embedding_avg=post_attn_embedding_avg,
                             text_token_head_output_avg=text_token_head_output_avg,
                             last_obs=obs,
                             frame_index_to_use=frame_index_to_use,
                             task_range=task_range,
                             ), f)

    # for reconstructing prompts
    unembedder = model.PaliGemma.llm.embedder["input_embedding"].transpose()
    embedder = model.PaliGemma.llm.embedder["input_embedding"]
    norm_embedding_matrix = normalize(embedder)  # shape: (vocab_size, embedding_dim)

    all_ret = {}
    for filename in tqdm.tqdm(os.listdir(reconstruct_ret_dir)):
        filepath = os.path.join(reconstruct_ret_dir, filename)

        with open(filepath, "rb") as f:
            task_tensor = pickle.load(f)

        ret = []
        embeddings = []
        decoding_result = []

        prompt = decode(task_tensor["last_obs"].tokenized_prompt[0].tolist())
        print(prompt)
        print(task_tensor["last_obs"].tokenized_prompt[0].tolist())
        num_valid_token = task_tensor["last_obs"].tokenized_prompt_mask.sum()
        print("num_valid_token:", num_valid_token)

        for layer in range(num_reconstruct_layer):
            embeddings.append(task_tensor["hidden_states_avg"][layer, 256 * 3:256 * 3 + num_valid_token])
            norm_embedded_tokens = normalize(task_tensor["hidden_states_avg"][layer, 256 * 3:256 * 3 + num_valid_token])
            similarity = jnp.dot(norm_embedded_tokens, norm_embedding_matrix.T)  # shape: (num_tokens, vocab_size)
            token_indices = jnp.argmax(similarity, axis=1)  # shape: (num_tokens,)
            if layer == 0:
                to_assert = token_indices[:num_valid_token] == task_tensor["last_obs"].tokenized_prompt[0,
                                                               :num_valid_token]
                assert jnp.all(to_assert)
            decoding_result.append(decode(token_indices[:num_valid_token].tolist()))
            ret.append(token_indices.tolist())
        all_ret[prompt[:-1].replace(" ", "_")] = {"token_indices": ret, "decoding_result": decoding_result}

    with open(os.path.join(reconstruct_ret_dir, "reconstruct_prompt.pkl"), "wb") as f:
        pickle.dump(all_ret, f)
