gcloud alpha compute tpus tpu-vm ssh wang-v4-32 --zone=us-central2-b --worker=all --command="\
export WANDB_API_KEY=$WANDB_API_KEY; \
cd EasyLM_run; \
git pull; \
export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'; \
pip install -r requirements.txt; \
python -m EasyLM.models.llama.llama_train_ppo \
    --mesh_dim='1,64,4' \
    --load_llama_config_policy='13b' \
    --load_llama_config_reward='13b' \
    --load_checkpoint_policy='params::gs://tdmpc-bucket/llama-1b/llama-1b.stream' \
    --load_checkpoint_reward='gs://tdmpc-bucket/grm-llama3.2-3b/rm_weights.stream' \
    --tokenizer.vocab_file='gs://tdmpc-bucket/grm-llama3.2-3b/tokenizer.json'\
    --tokenizer.add_bos_token=True \
    --train_dataset.type='tulu_prompt' \
    --train_dataset.tulu_prompt_dataset.path='gs://tdmpc-bucket/data/tulu-2.5-preference-data_ultrafeedback_mean_aspects.jsonl' \
    --train_dataset.tulu_prompt_dataset.seq_length=1024 \
    --max_continuation_len=1024 \
    --train_dataset.tulu_prompt_dataset.batch_size=1 \
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
    --logger.online=True \
    --logger.entity='nathom' \
    --logger.project='tdmpc-lm' \
    --logger.prefix='test_run' \
    --logger.prefix_to_id=True \
    --logger.wandb_dir='/home/ucsdwanglab/wandb' \
    --logger.output_dir='gs://tdmpc-bucket/n-tulu-ppo-jax/' \
    --use_tpu=True \
    --ppo_epochs=1 \
    --lr=1e-6 \
    --kl_coef=0.05 \
    --reward_gain=1.0 --reward_bias=0.0 \
    --save_milestone_freq=10000 \
    --num_epochs=1 \
    --max_steps_per_epoch=0 \
    --generate_only=False \
    | tee /home/ucsdwanglab/all.log \
"