# Paper: Unpacking DPO and PPO

This document is intended to provide some notes on how this library was used for the recent paper [Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback](https://arxiv.org/abs/2406.09279).

## Setup

Please see `docs/ai2.md` for general notes on setting up this repository and tips on how to use it. For people outside of Ai2, you'll need access to TPUs (potentially via the [TRC program](https://sites.research.google/trc/publications/)), a google bucket, and some GPUs for evaluation. I ran experiments on v3-128 and v3-256 TPUs. You probably can't go smaller than v3-128 for 13B models.

Especially note how commands are run on TPUs - I will assume the below commands are being run on setup TPUs via a command like:
```bash
gcloud alpha compute tpus tpu-vm ssh <name> --zone=us-east1-d --project=ai2-tpu --worker=all --command="<your command here>"
```

If you are new to TPUs, please read `ai2.md` carefully, and I'd advise some time playing around with the codebase and TPUs a bit before jumping into things like PPO training (which has a lot more moving parts).

## Data

You can find the datasets used [here for preference data](https://huggingface.co/datasets/allenai/tulu-2.5-preference-data) and [here for prompts](https://huggingface.co/datasets/allenai/tulu-2.5-prompts). You can load these direct from HF or from a google bucket path (in jsonl format).

## Models

Convert [Tulu 2 13B](https://huggingface.co/allenai/tulu-2-13b) to EasyLM format using `convert_hf_to_easylm`, and upload to your google bucket.

## DPO Experiments

Given a TPU, you can run a DPO experiment as follows (omitting the `tpu-vm ssh` command, see `ai2.md#Setup`):
```bash
cd easylm; git pull; export HF_TOKEN=<your_token_here>; export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'; python3 -m EasyLM.models.llama.llama_train_dpo \
    --mesh_dim='-1,16,16' \
    --dtype='bf16' \
    --num_epochs=3 \
    --log_freq=50 \
    --save_model_freq=1000 \
    --save_milestone_freq=0 \
    --load_llama_config='13b' \
    --update_llama_config='' \
    --load_dataset_state='' \
    --load_checkpoint='params::gs://<path_to_tulu_2_13b_weights>' \
    --tokenizer='allenai/tulu-2-13b' \
    --tokenizer_pad_token_id=0 \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=0.0 \
    --optimizer.adamw_optimizer.lr=5e-7 \
    --optimizer.adamw_optimizer.end_lr=0 \
    --optimizer.adamw_optimizer.warmup_ratio=0.1 \
    --optimizer.accumulate_gradient_steps=4 \
    --train_dataset.type='preference_json_torch' \
    --train_dataset.json_torch_dataset.hf_name='allenai/tulu-2.5-preference-data' \
    --train_dataset.json_torch_dataset.hf_split='ultrafeedback_mean_aspects' \
    --train_dataset.json_torch_dataset.seq_length=4096 \
    --train_dataset.json_torch_dataset.batch_size=8 \
    --checkpointer.save_optimizer_state=False \
    --logger.online=True --logger.project='<wandb_project>' --logger.entity='<wandb group>' \
    --logger.output_dir="gs://OUTPUT_DIR" &> all.log &"
```

Replace `hf_name` and `hf_split` with the split/HF dataset you want to use, or use `path` instead to point to google bucket files. You can check on your experiment in wandb or with `tail -f all.log` (on the TPU).

## PPO Experiments

For PPO experiments, we first need to train an RM, and then do PPO training.

First, to train an RM we can run:
```bash
cd easylm; git pull; export HF_TOKEN=<your_token_here>; export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'; python3 -m EasyLM.models.llama.llama_train_rm \
    --mesh_dim=-1,64,8 \
    --dtype=bf16 \
    --num_epochs=1 \
    --log_freq=50 \
    --save_model_freq=1000 \
    --save_milestone_freq=0 \
    --load_llama_config="13b" \
    --load_checkpoint='params::gs://<path_to_tulu_2_13b_weights>' \
    --tokenizer='allenai/tulu-2-13b' \
    --tokenizer_pad_token_id=0 \
    --optimizer.type=adamw \
    --optimizer.adamw_optimizer.weight_decay=0.0 \
    --optimizer.adamw_optimizer.lr=1e-5 \
    --optimizer.adamw_optimizer.end_lr=1e-6 \
    --optimizer.adamw_optimizer.warmup_ratio=0.03 \
    --optimizer.accumulate_gradient_steps=16 \
    --train_dataset.type=preference_json_torch \
    --train_dataset.json_torch_dataset.hf_name='allenai/tulu-2.5-preference-data' \
    --train_dataset.json_torch_dataset.hf_split='ultrafeedback_mean_aspects' \
    --train_dataset.json_torch_dataset.seq_length=4096 \
    --train_dataset.json_torch_dataset.batch_size=32 \
    --checkpointer.save_optimizer_state=False \
    --train_dataset.json_torch_dataset.remove_truncated_samples=True \
    --logger.online=True --logger.project='<wandb_project>' --logger.entity='<wandb group>' \
    --logger.output_dir="gs://OUTPUT_DIR" &> all.log &"
```

Next, grab the path to the final weights of your trained RM, and we can run PPO with the following command:
```bash
```bash
cd easylm; git pull; export HF_TOKEN=<your_token_here>; export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'; python3 -m EasyLM.models.llama.llama_train_ppo \
    --mesh_dim='1,64,4' \
    --load_llama_config_policy='13b' \
    --load_llama_config_reward='13b' \
    --load_checkpoint_policy='params::gs://<TULU_2_13B_PATH>' \
    --load_checkpoint_reward='params::gs://<REWARD_MODEL_PATH>' \
    --tokenizer='meta-llama/Llama-2-7b-hf' \
    --tokenizer_pad_token_id=0 \
    --tokenizer.add_bos_token=True \
    --train_dataset.type='tulu_prompt' \
    --train_dataset.tulu_prompt_dataset.hf_name='allenai/tulu-2.5-prompts' \
    --train_dataset.tulu_prompt_dataset.hf_split='ultrafeedback_prompts' \
    --train_dataset.tulu_prompt_dataset.seq_length=1024 \
    --max_continuation_len=1024 \
    --train_dataset.tulu_prompt_dataset.batch_size=512 \
    --rollouts_per_prompt=1 \
    --mini_batch_size=64 \
    --train_dataset.tulu_prompt_dataset.num_workers=16 \
    --train_dataset.tulu_prompt_dataset.remove_truncated_samples=True \
    --optimizer.type='adamw' \
    --optimizer.accumulate_gradient_steps=1 \
    --optimizer.adamw_optimizer.weight_decay=0.0 \
    --warmup_epochs=0.1 \
    --policy_freeze_epochs=0.0 \
    --checkpointer.save_optimizer_state=False \
    --use_tpu=True \
    --ppo_epochs=1 \
    --lr=1e-6 \
    --kl_coef=0.05 \
    --reward_gain=1.0 --reward_bias=0.0 \
    --save_milestone_freq=116 \
    --num_epochs=1 \
    --logger.online=True --logger.prefix_to_id=True --logger.project='<wandb_project>' --logger.entity='<wandb group>' \
    --logger.output_dir="gs://OUTPUT_DIR" &> all.log &"
```

For both these experiments, replace `hf_name` and `hf_split` with the split/HF dataset you want to use, or use `path` instead to point to google bucket files. You can check on your experiment in wandb or with `tail -f all.log` (on the TPU).

## Evaluation

First, I converted the requisite model to HF pytorch format. This can be done with the `convert_easylm_to_hf.py` script.

For running evaluation, I then used the [Open-Instruct evaluation suite](https://github.com/allenai/open-instruct) - please see the open-instruct repository for details on how to use that. All evaluations used in the paper are implemented there.


## Conclusion

And that's it! All data used for the paper should be available in the HuggingFace datasets linked above. For hyperparameters used please see Appendix F of the paper, and Table 9 - the above commands use some default hyperparameters, but these do not match in all cases (e.g. using 0.0325 kl coefficient for training with larger reward models).

If you have further comments, issues, or questions, please reach out to me - `hamishiv[at]cs.washington.edu`.