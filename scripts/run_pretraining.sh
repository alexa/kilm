python tasks/pretraining.py \
--dataset_name 'src/data/wikipedia.py' \
--dataset_config_name wiki_para \
--cache_dir $2 \
--tokenizer_name facebook/bart-base \
--model_name_or_path facebook/bart-base \
--mlm_probability 0.3 \
--poisson_lambda 3.0 \
--do_train \
--do_eval \
--logging_steps 50 \
--save_steps 500 \
--eval_steps 500 \
--evaluation_strategy steps \
--load_best_model_at_end \
--save_total_limit 10 \
--learning_rate 5e-5 \
--num_train_epochs 10 \
--max_source_length 512 \
--max_target_length 640 \
--output_dir $1 \
--preprocessing_num_workers 96 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--gradient_accumulation_steps 128 \
--overwrite_output_dir \
--warmup_steps 0 \
--weight_decay 0.01 \
--sharded_ddp simple \
--task lm 
