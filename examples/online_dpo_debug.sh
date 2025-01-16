python3 -m EasyLM.models.llama.llama_train_online_dpo \
    --mesh_dim='1,1,1' \
    --load_llama_config_policy='debug' \
    --load_llama_config_reward='debug' \
    --load_checkpoint_policy='' \
    --load_checkpoint_reward='' \
    --tokenizer.vocab_file='gs://jiachengl-east1/tokenizer.model' \
    --tokenizer.add_bos_token=True \
    --train_dataset.type='tulu_prompt' \
    --train_dataset.tulu_prompt_dataset.path='gs://hamishi-east1/easylm/data/converted_pref_data/ultrafeedback_mean_aspects_cleaned.jsonl' \
    --train_dataset.tulu_prompt_dataset.seq_length=256 \
    --max_continuation_len=16 \
    --train_dataset.tulu_prompt_dataset.batch_size=1 \
    --mini_batch_size=2 \
    --train_dataset.tulu_prompt_dataset.num_workers=16 \
    --train_dataset.tulu_prompt_dataset.remove_truncated_samples=True \
    --optimizer.type='adamw' \
    --optimizer.accumulate_gradient_steps=1 \
    --optimizer.adamw_optimizer.weight_decay=0.0 \
    --warmup_epochs=0.1 \
    --checkpointer.save_optimizer_state=False \
    --logger.online=False \
    --logger.entity='liujch1998' \
    --logger.project='n-Tulu-DPO-Jax' \
    --logger.prefix='debug' \
    --logger.prefix_to_id=True \
    --logger.wandb_dir='tmp' \
    --logger.output_dir='tmp' \
    --use_tpu=False \
    --ppo_epochs=1 \
    --lr=5e-7 \
    --beta=0.1 \
    --save_milestone_freq=0 \
    --num_epochs=1 \
    --max_steps_per_epoch=0 \
    --generate_only=False