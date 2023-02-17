"""
This script is modified from HuggingFace [Trainer]
(https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L1699).

In the original Trainer, everytime when a checkpoint is saved, the states of the 
tokenizer and the optimizer will also be saved. If we are using ShardedDDP, since 
the optimizer states are sharded to avoid spending too much GPU memory, we get 
the OOM error when the Trainer tries to save the optimizer state.

This is related to the [issue](https://github.com/huggingface/transformers/issues/14542).
"""
import os
import random
import numpy as np
from packaging import version

import torch

from transformers import Trainer, Seq2SeqTrainer
from transformers.utils import logging
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    HPSearchBackend,
    ShardedDDPOption,
)
from transformers.file_utils import (
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
)

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast

logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"

class NewTrainer(Trainer):
    """
    With huggingface version == 4.12.3, there will be CUDA OOM error when saving the model checkpoints with
    multiple GPUs + sharded_ddp.
    Skip saving the optimizers and the schedulers for now.
    """
    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is not None and trial is not None:
            if self.hp_search_backend == HPSearchBackend.OPTUNA:
                run_id = trial.number
            elif self.hp_search_backend == HPSearchBackend.RAY:
                from ray import tune

                run_id = tune.get_trial_id()
            elif self.hp_search_backend == HPSearchBackend.SIGOPT:
                run_id = trial.id
            run_name = self.hp_name(trial) if self.hp_name is not None else f"run-{run_id}"
            run_dir = os.path.join(self.args.output_dir, run_name)
        else:
            run_dir = self.args.output_dir
            self.store_flos()

        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir)
        if self.deepspeed:
            # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
            # config `stage3_gather_fp16_weights_on_model_save` is True
            self.deepspeed.save_checkpoint(output_dir)

        """The following codes are commented out to skip saving the state of the optimizer."""
        # # Save optimizer and scheduler
        # if self.sharded_ddp == ShardedDDPOption.SIMPLE:
        #     self.optimizer.consolidate_state_dict()

        # if is_torch_tpu_available():
        #     xm.rendezvous("saving_optimizer_states")
        #     xm.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
        #     with warnings.catch_warnings(record=True) as caught_warnings:
        #         xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
        #         reissue_pt_warnings(caught_warnings)
        # elif is_sagemaker_mp_enabled():
        #     if smp.dp_rank() == 0:
        #         # Consolidate the state dict on all processed of dp_rank 0
        #         opt_state_dict = self.optimizer.state_dict()
        #         # Save it and the scheduler on the main process
        #         if self.args.should_save:
        #             torch.save(opt_state_dict, os.path.join(output_dir, OPTIMIZER_NAME))
        #             with warnings.catch_warnings(record=True) as caught_warnings:
        #                 torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
        #             reissue_pt_warnings(caught_warnings)
        #             if self.use_amp:
        #                 torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))
        # elif self.args.should_save and not self.deepspeed:
        #     # deepspeed.save_checkpoint above saves model/optim/sched
        #     torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
        #     with warnings.catch_warnings(record=True) as caught_warnings:
        #         torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
        #     reissue_pt_warnings(caught_warnings)
        #     if self.use_amp:
        #         torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            if self.args.local_rank == -1:
                # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
                rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
            else:
                rng_states["cuda"] = torch.cuda.random.get_rng_state()

        if is_torch_tpu_available():
            rng_states["xla"] = xm.get_rng_state()

        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)
        local_rank = xm.get_local_ordinal() if is_torch_tpu_available() else self.args.local_rank
        if local_rank == -1:
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            torch.save(rng_states, os.path.join(output_dir, f"rng_state_{local_rank}.pth"))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)


class NewSeq2SeqTrainer(Seq2SeqTrainer):
    """
    With huggingface version == 4.12.3, there will be CUDA OOM error when saving the model checkpoints with
    multiple GPUs + sharded_ddp.
    Skip saving the optimizers and the schedulers for now.
    """
    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is not None and trial is not None:
            if self.hp_search_backend == HPSearchBackend.OPTUNA:
                run_id = trial.number
            elif self.hp_search_backend == HPSearchBackend.RAY:
                from ray import tune

                run_id = tune.get_trial_id()
            elif self.hp_search_backend == HPSearchBackend.SIGOPT:
                run_id = trial.id
            run_name = self.hp_name(trial) if self.hp_name is not None else f"run-{run_id}"
            run_dir = os.path.join(self.args.output_dir, run_name)
        else:
            run_dir = self.args.output_dir
            self.store_flos()

        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir)
        if self.deepspeed:
            # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
            # config `stage3_gather_fp16_weights_on_model_save` is True
            self.deepspeed.save_checkpoint(output_dir)

        """The following code is commented out to skip saving the state of the optimizer."""
        # # Save optimizer and scheduler
        # if self.sharded_ddp == ShardedDDPOption.SIMPLE:
        #     self.optimizer.consolidate_state_dict()

        # if is_torch_tpu_available():
        #     xm.rendezvous("saving_optimizer_states")
        #     xm.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
        #     with warnings.catch_warnings(record=True) as caught_warnings:
        #         xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
        #         reissue_pt_warnings(caught_warnings)
        # elif is_sagemaker_mp_enabled():
        #     if smp.dp_rank() == 0:
        #         # Consolidate the state dict on all processed of dp_rank 0
        #         opt_state_dict = self.optimizer.state_dict()
        #         # Save it and the scheduler on the main process
        #         if self.args.should_save:
        #             torch.save(opt_state_dict, os.path.join(output_dir, OPTIMIZER_NAME))
        #             with warnings.catch_warnings(record=True) as caught_warnings:
        #                 torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
        #             reissue_pt_warnings(caught_warnings)
        #             if self.use_amp:
        #                 torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))
        # elif self.args.should_save and not self.deepspeed:
        #     # deepspeed.save_checkpoint above saves model/optim/sched
        #     torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
        #     with warnings.catch_warnings(record=True) as caught_warnings:
        #         torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
        #     reissue_pt_warnings(caught_warnings)
        #     if self.use_amp:
        #         torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            if self.args.local_rank == -1:
                # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
                rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
            else:
                rng_states["cuda"] = torch.cuda.random.get_rng_state()

        if is_torch_tpu_available():
            rng_states["xla"] = xm.get_rng_state()

        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)
        local_rank = xm.get_local_ordinal() if is_torch_tpu_available() else self.args.local_rank
        if local_rank == -1:
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            torch.save(rng_states, os.path.join(output_dir, f"rng_state_{local_rank}.pth"))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)
