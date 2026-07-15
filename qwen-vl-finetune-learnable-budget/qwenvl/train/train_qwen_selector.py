# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import qwenvl.train.trainer
from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)

from qwenvl.data.data_qwen import make_supervised_data_module
from qwenvl.data.data_qwen_packed import make_supervised_data_module_packed
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2RMSNorm, Qwen2_5_VLVisionFlashAttention2, Qwen2_5_VLVisionBlock
import torch.nn as nn
from compression_method.selector_scorer import TransformerScorer
from compression_method.selector_model import (
   qwen25vl_vision_tower_forward_selector,
   qwen25vl_generation_forward_selector 
)
import types
import random
import numpy as np

local_rank = None

class ScheduledWeightTrainer(Trainer):
    def __init__(self, *args, reg_weight_start=0.1, reg_weight_end=3.0, budget_lr=0.01, compression_weight_start=0.5, compression_weight_end=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_weight_start = reg_weight_start
        self.reg_weight_end = reg_weight_end
        self.budget_lr = budget_lr
        self.compression_weight_start = compression_weight_start
        self.compression_weight_end = compression_weight_end

    def create_optimizer(self):
        """
        Override to ensure the budget parameter gets its own optimizer group.
        DeepSpeed ZeRO-3 may not properly update 1-element persistent parameters
        unless they are explicitly in the optimizer with a dedicated learning rate.
        """
        import torch
        opt_model = self.model_wrapped if hasattr(self, 'model_wrapped') else self.model

        # Separate budget parameter from other parameters
        budget_params = []
        other_params = []
        decay_parameters = self.get_decay_parameter_names(opt_model)

        for n, p in opt_model.named_parameters():
            if not p.requires_grad:
                continue
            if 'budget' in n:
                budget_params.append(p)
            else:
                other_params.append((n, p))

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in other_params if n in decay_parameters],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in other_params if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]
        # Add budget parameters with their own learning rate and no weight decay
        if budget_params:
            optimizer_grouped_parameters.append({
                "params": budget_params,
                "lr": self.budget_lr,
                "weight_decay": 0.0,
            })

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Overrides compute_loss to dynamically calculate regularization_weight.
        """
        total_steps = self.state.max_steps
        current_step = self.state.global_step

        if total_steps > 0:
            # Use min to ensure progress does not exceed 1.0
            progress = min(current_step / total_steps, 1.0)
            current_weight = self.reg_weight_start + (self.reg_weight_end - self.reg_weight_start) * progress
            current_compression_weight = self.compression_weight_start + (self.compression_weight_end - self.compression_weight_start) * progress
        else:
            # Use the starting weight if total_steps is not yet computed (value is -1)
            current_weight = self.reg_weight_start
            current_compression_weight = self.compression_weight_start

        # Set the calculated weights on the actual model
        actual_model = model.module if hasattr(model, 'module') else model
        actual_model.regularization_weight = current_weight
        actual_model.compression_weight = current_compression_weight

        # Call the parent's compute_loss method
        outputs = super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

        # Log individual loss components
        if self.is_world_process_zero() and self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
            task_loss = getattr(actual_model, 'task_loss', 0.0)
            bce_loss = getattr(actual_model, 'bce_loss', 0.0)
            comp_loss = getattr(actual_model, 'comp_loss', 0.0)
            total_loss = getattr(actual_model, 'total_loss', 0.0)
            budget_val = getattr(actual_model, 'current_budget_value', None)
            if budget_val is not None:
                print(f"[Step {self.state.global_step}] total_loss: {total_loss:.5f}, task_loss: {task_loss:.5f}, bce_loss: {bce_loss:.5f}, comp_loss: {comp_loss:.5f}, budget: {budget_val:.4f}, reg_weight: {current_weight:.4f}, comp_weight: {current_compression_weight:.4f}")
            else:
                print(f"[Step {self.state.global_step}] total_loss: {total_loss:.5f}, task_loss: {task_loss:.5f}, bce_loss: {bce_loss:.5f}, reg_weight: {current_weight:.4f}")
            try:
                import wandb
                if wandb.run is not None:
                    log_dict = {
                        'task_loss': task_loss,
                        'bce_loss': bce_loss,
                        'comp_loss': comp_loss,
                        'total_loss': total_loss,
                        'reg_weight': current_weight,
                        'compression_weight': current_compression_weight,
                    }
                    if budget_val is not None:
                        log_dict['budget_value'] = budget_val
                        # Also log the raw budget parameter (before clamping)
                        raw_budget = getattr(actual_model.visual.importance_scorer, 'budget', None)
                        if raw_budget is not None:
                            log_dict['budget_raw'] = raw_budget.data.item()
                    wandb.log(log_dict, step=self.state.global_step)
            except ImportError:
                pass

        return outputs



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False
    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False
    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        for n, p in model.lm_head.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        for n, p in model.lm_head.named_parameters():
            p.requires_grad = False
    # -------------------------add compressor tuning---------------------------------
    if model_args.tune_compressor:
        for n, p in model.visual.importance_scorer.named_parameters():
            p.requires_grad = True
        # budget is inside importance_scorer, so it's already handled by the loop above
    else:
        for n, p in model.visual.importance_scorer.named_parameters():
            p.requires_grad = False
    # -------------------------------------------------------------------------------

def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print('seed:', training_args.seed)
    print('data_seed:', training_args.data_seed)
    set_seed(training_args.seed)

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor

        data_args.model_type = "qwen2.5vl"
    else:
        raise ValueError("Model not currently supported")
        
    # ----------------------------add compressor setup-------------------------------
    # Learnable budget: passed to TransformerScorer as init_budget, stored as nn.Parameter inside scorer
    model.visual.forward = types.MethodType(qwen25vl_vision_tower_forward_selector, model.visual)
    if "3b" in model_args.model_name_or_path.lower():
        print("3b")
        model.visual.importance_scorer = TransformerScorer(in_features=2048, num_kv_heads=2, intermediate_size=11008, init_budget=model_args.budget)
    elif "7b" in model_args.model_name_or_path.lower():
        print("7b")
        model.visual.importance_scorer = TransformerScorer(in_features=3584, num_kv_heads=4, intermediate_size=18944, init_budget=model_args.budget)
    else:
        raise ValueError("Model not currently supported")

    # Load pretrained LLM middle layer weights for scorer initialization
    scorer = model.visual.importance_scorer
    if "3b" in model_args.model_name_or_path.lower():
        scorer_init_path = project_root / "compression_method" / "scorer_init_3b.pt"
    elif "7b" in model_args.model_name_or_path.lower():
        scorer_init_path = project_root / "compression_method" / "scorer_init_7b.pt"
    else:
        raise ValueError("Model not currently supported")

    if scorer_init_path.exists():
        init_weights = torch.load(str(scorer_init_path), map_location="cpu")
        with torch.no_grad():
            scorer.q_proj.weight.copy_(init_weights["q_proj.weight"])
            scorer.q_proj.bias.copy_(init_weights["q_proj.bias"])
            scorer.k_proj.weight.copy_(init_weights["k_proj.weight"])
            scorer.k_proj.bias.copy_(init_weights["k_proj.bias"])
            scorer.input_layernorm.weight.copy_(init_weights["input_layernorm.weight"])
        print(f"Loaded scorer init weights from {scorer_init_path}")
    else:
        print(f"Warning: scorer init weights not found at {scorer_init_path}, using random init")

    model.forward = types.MethodType(qwen25vl_generation_forward_selector, model)

    # Initialize compression_weight for compression loss (will be updated by curriculum learning in compute_loss)
    model.compression_weight = training_args.compression_weight_start
    print(f"Initial budget value: {model.visual.importance_scorer.budget.item():.4f}")
    print(f"Compression weight: {training_args.compression_weight_start:.4f} -> {training_args.compression_weight_end:.4f} (curriculum learning)")
    # -------------------------------------------------------------------------------

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False


    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    set_model(model_args, model)

    # --- print trainable parameters ---
    if local_rank == 0: 
        print("="*80)
        print("Printing trainable parameters...")
        trainable_param_names = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_param_names.append(name)
        
        for name in trainable_param_names:
            print(f"- {name}")
        print("="*80)
    # -----------------------------------

    if torch.distributed.get_rank() == 0:
        model.visual.print_trainable_parameters()
        model.model.print_trainable_parameters()
    
    if data_args.data_packing:
        data_module = make_supervised_data_module_packed(tokenizer=tokenizer, data_args=data_args)
    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = ScheduledWeightTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        reg_weight_start=training_args.reg_weight_start,  # Set starting weight
        reg_weight_end=training_args.reg_weight_end,    # Set ending weight
        budget_lr=training_args.budget_lr,              # Learning rate for budget parameter
        compression_weight_start=training_args.compression_weight_start,  # Starting compression weight
        compression_weight_end=training_args.compression_weight_end,    # Ending compression weight
        **data_module
    )

    # trainer = Trainer(
    #     model=model, processing_class=tokenizer, args=training_args, **data_module
    # )


    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # After training, copy preprocessor_config.json and chat_template.json
    if local_rank == 0:
        source_dir = pathlib.Path(model_args.model_name_or_path)
        dest_dir = pathlib.Path(training_args.output_dir)
        
        preprocessor_file = "preprocessor_config.json"
        chat_template_file = "chat_template.json"
        
        # Copy preprocessor_config.json
        source_preprocessor_path = source_dir / preprocessor_file
        dest_preprocessor_path = dest_dir / preprocessor_file
        if source_preprocessor_path.exists():
            shutil.copy(source_preprocessor_path, dest_preprocessor_path)
            rank0_print(f"Copied {source_preprocessor_path} to {dest_preprocessor_path}")
        else:
            rank0_print(f"Warning: {source_preprocessor_path} not found.")

        # Copy chat_template.json
        source_chat_template_path = source_dir / chat_template_file
        dest_chat_template_path = dest_dir / chat_template_file
        if source_chat_template_path.exists():
            shutil.copy(source_chat_template_path, dest_chat_template_path)
            rank0_print(f"Copied {source_chat_template_path} to {dest_chat_template_path}")
        else:
            rank0_print(f"Warning: {source_chat_template_path} not found.")


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
