# Adopted from https://github.com/haotian-liu/LLaVA    . Below is the original copyright:
# Adopted from https://github.com/lm-sys/FastChat    . Below is the original copyright:
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


import re
import os
import sys
import copy
import json
import random
import pathlib
import traceback
import math
import argparse
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

# torch-related packages
# NOTE: torch must be imported before transformers. Otherwise, `Segmentation fault (core dumped)` will occur.
import torch
from torch.utils.data import Dataset

import transformers
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(BASE_DIR)
# sys.path.append('./motif/motif')
from motif.model import *
from motif.constants import NUM_FRAMES, IGNORE_INDEX, MODAL_INDEX_MAP
from motif.mm_utils import tokenizer_multimodal_token, process_video, process_image
from motif.videollama2_trainer import (VideoLLaMA2Trainer,
    get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, 
    find_all_linear_names, safe_save_model_for_hf_trainer
)

# NOTE: fast tokenizer warning issue: https://github.com/huggingface/transformers/issues/5486       
os.environ["TOKENIZERS_PARALLELISM"] = "true"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def set_seed(seed=42):
    """
    Set the random seed for reproducible results.

    :param seed: An integer value to be used as the random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class ModelArguments:
    # LLM Arguments
    model_type: Optional[str] = field(default="videollama2", metadata={"help": "Model type selected in the list: " + ", ".join(VLLMs.keys())})
    model_path: Optional[str] = field(default="lmsys/vicuna-7b-v1.5")
    version: Optional[str] = field(default="v1", metadata={"help": "Version of the conversation template."})
    freeze_backbone: bool = field(default=False, metadata={"help": "Whether to freeze the LLM backbone."})
    # Connector Arguments
    mm_projector_type: Optional[str] = field(default='linear')
    tune_mm_mlp_adapter: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    # Vision tower Arguments
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    # Path Arguments
    data_path: List[str] = field(default=None, metadata={"help": "Path to the training data."})
    # image_folder: Optional[str] = field(default=None)
    # video_folder: Optional[str] = field(default=None)
    data_folder: Optional[str] = field(default=None)
    # Loading Arguments
    is_multimodal: bool = False
    lazy_preprocess: bool = False
    num_frames: Optional[int] = field(default=None)
    # Preprocess Arguments
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    mem_monitor: bool = field(default=False, metadata={"help": "Enable memory monitoring thread"})
    mm_projector_lr: Optional[float] = None
    freeze_mm_mlp_adapter: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)
    # Training Data Arguments 
    group_by_modality_length: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    # Lora or Quant Arguments
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    modal_token: str = None,
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    conversations = []
    input_ids = []
    targets = []
    for source in sources:
        # 1. apply chat template for input conversation
        assert len(source) == 2
        assert modal_token in source[0]['value']
        message = [
            {'role': 'user', 'content': modal_token},
            {'role': 'assistant', 'content': source[1]['value']}
        ]
        conversation = " ".join([sentence['value'] for sentence in source])

        input_id = tokenizer_multimodal_token(conversation, tokenizer, modal_token, return_tensors='pt')
        target = copy.deepcopy(input_id)
        target[input_id == MODAL_INDEX_MAP[modal_token]] = IGNORE_INDEX

        input_ids.append(input_id)
        targets.append(target)

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    modal_token: str = None,
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}

    # Apply prompt templates
    conversations = []
    input_ids = []
    targets = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != "user":
            # Skip the first one if it is not from human
            source = source[1:]

        message = [{'role': roles[sentence['from']], 'content': sentence['value']} for sentence in source]
        conversation = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
        input_ids.append(tokenizer_multimodal_token(conversation, tokenizer, modal_token, return_tensors='pt'))
        targets.append(copy.deepcopy(input_ids[-1]))

        assert len(source) % 2 == 0, f"Invalid conversation length {len(source)}."

        cur = 0
        message = []
        for idx, sentence in enumerate(source):
            if idx % 2 == 1:
                tmp_message = [
                    {'role': roles[source[idx-1]['from']], 'content': source[idx-1]['value']}, 
                    {'role': roles[sentence['from']], 'content': sentence['value']}
                ]

                instruction = tokenizer.apply_chat_template(message + tmp_message[:1], tokenize=False, add_generation_prompt=True)
                conversation = tokenizer.apply_chat_template(message + tmp_message, tokenize=False, add_generation_prompt=False)

                instruction_len = len(tokenizer_multimodal_token(instruction, tokenizer, modal_token, return_tensors='pt'))
                conversation_len = len(tokenizer_multimodal_token(conversation, tokenizer, modal_token, return_tensors='pt'))

                targets[-1][cur:instruction_len] = IGNORE_INDEX

                cur = conversation_len
                message += tmp_message

    return dict(input_ids=input_ids, labels=targets)


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments,
    modal_token: str = None,
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    assert modal_token in MODAL_INDEX_MAP, f"Unsupported modal token {modal_token}."

    for source in sources:
        for sentence in source:
            if modal_token in sentence['value']:
                sentence['value'] = sentence['value'].replace(modal_token, '').strip()
                sentence['value'] = modal_token + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
            replace_token = modal_token
            # TODO: fix this for multimedia, e.g., <video>, <audio>, etc.
            sentence["value"] = sentence["value"].replace(modal_token, replace_token)

    return sources


class MemoryCleanCallback(transformers.TrainerCallback):
    """Callback to clean memory and reset peak stats every N steps"""
    def __init__(self, clean_interval=100):
        self.clean_interval = clean_interval
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.clean_interval == 0:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            if local_rank in [-1, 0]:
                mem_used = torch.cuda.memory_allocated() / 1024**3
                mem_reserved = torch.cuda.memory_reserved() / 1024**3
                rank0_print(f"[Memory Clean] Step {state.global_step}: Used={mem_used:.2f}GB, Reserved={mem_reserved:.2f}GB")


class MemoryMonitorThread(threading.Thread):
    """Background thread for monitoring GPU memory usage"""
    def __init__(self, interval=60):
        super().__init__(daemon=True)
        self.interval = interval
        self.running = True
    
    def run(self):
        while self.running:
            if local_rank in [-1, 0]:
                mem_used = torch.cuda.memory_allocated() / 1024**3
                mem_reserved = torch.cuda.memory_reserved() / 1024**3
                peak_used = torch.cuda.max_memory_allocated() / 1024**3
                rank0_print(f"[Memory Monitor] Used: {mem_used:.2f}GB, Reserved: {mem_reserved:.2f}GB, Peak: {peak_used:.2f}GB")
            time.sleep(self.interval)
    
    def stop(self):
        self.running = False


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = []
        for dp in data_path:
            _datas = json.load(open(dp, "r"))
            list_data_dict.extend(_datas)

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 576 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        image_processor = self.data_args.image_processor
        video_processor = self.data_args.video_processor

        num_frames = NUM_FRAMES if self.data_args.num_frames is None else self.data_args.num_frames

        max_retries = 3
        for attempt in range(max_retries):
            try:
                if 'image' in sources[0]:
                    image_file = self.list_data_dict[i]['image']
                    image_folder = self.data_args.data_folder
                    image_file = os.path.join(image_folder, image_file)

                    image = process_image(image_file, image_processor, aspect_ratio=self.data_args.image_aspect_ratio)

                    # place <image> tag to question head.
                    modal_token = "<image>"
                    sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args, modal_token)
                elif 'video' in sources[0]:
                    video_file = self.list_data_dict[i]['video']
                    video_folder = self.data_args.data_folder
                    video_file = os.path.join(video_folder, video_file)

                    # 优化视频帧处理：动态调整帧数
                    video = process_video(video_file, video_processor, aspect_ratio=self.data_args.image_aspect_ratio, num_frames=num_frames)

                    # place <video> tag to question head.
                    modal_token = "<video>"
                    sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args, modal_token)
                else:
                    modal_token = None
                    sources = copy.deepcopy([e["conversations"] for e in sources])

                if self.data_args.is_pretraining:
                    data_dict = preprocess_plain(sources, self.tokenizer, modal_token=modal_token)
                else:
                    data_dict = preprocess(sources, self.tokenizer, modal_token=modal_token)

                if isinstance(i, int):
                    data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

                # image exist in the data
                if 'image' in self.list_data_dict[i]:
                    data_dict['image'] = image
                elif 'video' in self.list_data_dict[i]:
                    data_dict['video'] = video
                elif self.data_args.is_multimodal:
                    # image does not exist in the data, but the model is multimodal
                    data_dict['image'] = torch.zeros(3, self.data_args.image_size, self.data_args.image_size)
                return data_dict
            except Exception as e:
                if attempt < max_retries - 1:
                    backup_idx = random.randint(0, len(self.list_data_dict) - 1)
                    rank0_print(f"Attempt {attempt+1} failed for index {i}: {str(e)}. Trying backup index {backup_idx}")
                    i = backup_idx
                    sources = self.list_data_dict[i]
                else:
                    traceback.print_exc()
                    raise RuntimeError(f"Failed to load sample after {max_retries} attempts")

        # Should never reach here
        raise RuntimeError("Unexpected error in __getitem__")


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        # work for 'images' argument in `prepare_inputs_labels_for_multimodal` of LlavaMetaForCausalLM in llava_arch.py
        batch['images'] = []
        for instance in instances:
            for modal_token in MODAL_INDEX_MAP.keys():
                modal_token = modal_token.lower()
                # MODAL_TOKEN shape like: <image>, <video>, ...
                modal_name = re.findall(f'[<](.*)[>]', modal_token)
                assert len(modal_name) == 1
                modal_name = modal_name[0]
                if modal_name in instance:
                    batch['images'].append((instance[modal_name], modal_name))

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


class MemoryEfficientGradientAccumulator(torch.optim.Optimizer):
    """流式梯度累积优化器，减少中间状态存储"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.98)):
        defaults = dict(lr=lr, betas=betas)
        super().__init__(params, defaults)
        
        # 初始化状态
        self.state = defaultdict(dict)
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['step'] = 0
                self.state[p]['exp_avg'] = torch.zeros_like(p.data)
                self.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
    
    def step(self, closure=None):
        """执行部分参数更新"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # 更新状态
                state = self.state[p]
                state['step'] += 1
                
                # 指数移动平均
                state['exp_avg'] = beta1 * state['exp_avg'] + (1 - beta1) * grad
                state['exp_avg_sq'] = beta2 * state['exp_avg_sq'] + (1 - beta2) * grad**2
                
                # 偏置校正
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # 计算更新量
                denom = (state['exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)) + 1e-8
                step_size = lr / bias_correction1
                
                # 应用部分更新
                p.data.addcdiv_(state['exp_avg'], denom, value=-step_size)
        
        # 清空梯度
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
        
        return loss


def train(attn_implementation=None):
    global local_rank
    set_seed(42)

    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description="VideoLLaMA2 Training Script", add_help=False)
    
    # 添加自定义参数
    parser.add_argument("--optim", type=str, default="adamw_torch", 
                       choices=["adamw_torch", "mem_eff_adam"],
                       help="Optimizer type")
    parser.add_argument("--mem_monitor", action="store_true", 
                       help="Enable memory monitoring thread")
    
    # 解析已知的自定义参数
    custom_args, remaining_args = parser.parse_known_args()
    
    # 解析Hugging Face参数
    hf_parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = hf_parser.parse_args_into_dataclasses(remaining_args)

    # 应用自定义参数到训练参数
    training_args.optim = custom_args.optim
    training_args.mem_monitor = custom_args.mem_monitor

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type, # {'fp4', 'nf4'}
                bnb_4bit_quant_storage=compute_dtype,
            )
        ))

    config = VLLMConfigs[model_args.model_type].from_pretrained(model_args.model_path, trust_remote_code=True)
    config._attn_implementation = attn_implementation
    config.use_cache = False  # 确保训练时关闭缓存

    if model_args.vision_tower is not None:
        model = VLLMs[model_args.model_type].from_pretrained(
            model_args.model_path,
            config=config,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            do_sample=True,
            **bnb_model_from_pretrained_args
        )
        if 'mixtral' in model_args.model_type:
            import deepspeed
            deepspeed.utils.set_z3_leaf_modules(model, [MixtralSparseMoeBlock])
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_path,
            config=config,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            do_sample=True,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)


    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    if model_args.vision_tower is not None:
        # initialize vision encoder + multi-modal projector
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_size = vision_tower.image_size

        data_args.image_processor = vision_tower.image_processor
        data_args.video_processor = vision_tower.video_processor if hasattr(vision_tower, "video_processor") else vision_tower.image_processor

        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        if model_args.tune_mm_mlp_adapter:
            data_args.is_pretraining = True
        else:
            data_args.is_pretraining = False

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.num_frames = NUM_FRAMES if data_args.num_frames is None else data_args.num_frames

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # ====== 关键优化：内存高效优化器 ======
    optimizer = None
    if training_args.optim == "mem_eff_adam":
        rank0_print("Using memory efficient gradient accumulator")
        # 确保梯度累积步数合理
        if training_args.gradient_accumulation_steps <= 1:
            training_args.gradient_accumulation_steps = 8
            rank0_print(f"Setting gradient accumulation steps to {training_args.gradient_accumulation_steps} for memory efficient optimizer.")
        
        optimizer = MemoryEfficientGradientAccumulator(
            model.parameters(),
            lr=training_args.learning_rate,
            betas=(training_args.adam_beta1, training_args.adam_beta2)
        )
        
        # 覆盖训练参数
        training_args.optimizer = optimizer
        training_args.optim = "custom"
    # ====== 结束优化器设置 ======
    
    rank0_print("Current model:", model)
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    # 创建训练器
    trainer = VideoLLaMA2Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    
    # 添加显存清理回调
    trainer.add_callback(MemoryCleanCallback(clean_interval=50))
    
    # 启动显存监控线程
    mem_monitor = None
    if training_args.mem_monitor and local_rank in [-1, 0]:
        mem_monitor = MemoryMonitorThread(interval=60)
        mem_monitor.start()
        rank0_print("Memory monitoring thread started")

    # 训练模型
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    # 停止显存监控
    if mem_monitor:
        mem_monitor.stop()
        mem_monitor.join()


if __name__ == "__main__":
    train("flash_attention_2")