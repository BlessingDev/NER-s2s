
from typing import Optional, Union, Any
import numpy as np
import time
import math
import contextlib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
from torch.distributed.fsdp import FullyShardedDataParallel

from transformers import Seq2SeqTrainer
from datasets import Dataset
from transformers.trainer_utils import speed_metrics
from transformers.utils import (
    is_sagemaker_mp_enabled,
)
from transformers.trainer_pt_utils import (
    find_batch_size, 
    IterableDatasetShard, 
    EvalLoopContainer,
    nested_detach,
)
from transformers import logging
from transformers.trainer_utils import has_length, denumpify_detensorize, EvalLoopOutput, EvalPrediction

from transformers.integrations.deepspeed import deepspeed_init

import random

logger = logging.get_logger(__name__)

class ContrastiveGroupedBatchSampler(Sampler):
    def __init__(self, dataset, group_ids, batch_size):
        self.dataset = dataset
        self.group_ids = group_ids # List of group identifiers for each sample
        self.batch_size = batch_size
        self.groups = self._create_groups()
        
        self.batch_num = math.ceil(len(dataset) / batch_size)

    def _create_groups(self):
        # Logic to group indices by group_id
        # Example: {group1_id: [idx1, idx2], group2_id: [idx3, idx4, idx5]}
        groups = {}
        for i, group_id in enumerate(self.group_ids):
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(i)
        return groups

    def _get_curriculum_indices(self):
        shuffled_group_keys = list(self.groups.keys())
        #torch.manual_seed(torch.initial_seed()) # Ensure reproducibility if needed
        random.shuffle(shuffled_group_keys)
        max_id = max(self.groups.keys())

        all_indices = []
        for group_key in shuffled_group_keys:
            if group_key != max_id:
                # 마지막 그룹은 제외하고 섞기
                group_indices = self.groups[group_key]
                # Shuffle within each group
                #random.shuffle(group_indices)
                all_indices.extend(group_indices)
        
        # 마지막 그룹은 맨 뒤에 추가
        group_indices = self.groups[max_id]
        random.shuffle(group_indices)
        all_indices.extend(group_indices)
        
        return all_indices
    
    def _get_contrastive_indices(self):
        shuffled_group_keys = list(self.groups.keys())
        random.shuffle(shuffled_group_keys)

        all_group_keys = []
        all_indices = []
        for group_key in shuffled_group_keys:
            group_indices = self.groups[group_key]
            # Shuffle within each group
            #random.shuffle(group_indices)
            all_indices.extend(group_indices)
            all_group_keys.extend([group_key] * len(group_indices))
        
        return all_indices, all_group_keys
    
    def __iter__(self):
        # Shuffle groups themselves
        all_indices, all_group_keys = self._get_contrastive_indices()
        
        # Create batches from the shuffled indices
        # This part requires careful implementation to ensure batch_size respected
        # and potentially handling uneven group sizes
        start_idx = 0
        batch_idx = 0
        while start_idx < len(all_indices):
            cur_start_idx = start_idx
            start_idx += self.batch_size
            
            # 미니 배치에서 그룹이 잘리지 않도록 처리하기
            # 현재까지 진행된 인덱스가 본래 배치의 진행 상황과 어떻게 차이가 나는지 확인
            if start_idx < len(all_indices) and all_group_keys[start_idx - 1] == all_group_keys[start_idx]:
                # 배치 내 마지막 그룹 idx와 배치 바로 밖 그룹 idx가 같을 경우
                # 현재 배치를 해당 그룹을 포함하여 늘리거나, 현재 배치에서 해당 그룹을 제거하여 크기를 줄여야 한다.
                batch_progression = (batch_idx + 1) * self.batch_size - 1
                if batch_progression >= start_idx:
                    # 현재 그룹을 포함하여 배치를 늘리는 경우
                    while start_idx < len(all_indices) and all_group_keys[start_idx] == all_group_keys[start_idx - 1]:
                        start_idx += 1
                else:
                    # 현재 그룹을 제외하여 배치를 줄이는 경우
                    while start_idx > cur_start_idx and all_group_keys[start_idx - 1] == all_group_keys[start_idx]:
                        start_idx -= 1
            
            cur_batch = all_indices[cur_start_idx:start_idx]
            batch_idx += 1
            yield cur_batch

    def __len__(self):
        # Calculate the total number of batches
        return self.batch_num

class CurriculumGroupedBatchSampler(Sampler):
    def __init__(self, dataset, group_ids, batch_size):
        self.dataset = dataset
        self.group_ids = group_ids # List of group identifiers for each sample
        self.batch_size = batch_size
        self.groups = self._create_groups()
        
        self.batch_num = math.ceil(len(dataset) / batch_size)

    def _create_groups(self):
        # Logic to group indices by group_id
        # Example: {group1_id: [idx1, idx2], group2_id: [idx3, idx4, idx5]}
        groups = {}
        for i, group_id in enumerate(self.group_ids):
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(i)
        return groups

    def _get_curriculum_indices(self):
        shuffled_group_keys = list(self.groups.keys())
        #torch.manual_seed(torch.initial_seed()) # Ensure reproducibility if needed
        random.shuffle(shuffled_group_keys)
        max_id = max(self.groups.keys())

        all_indices = []
        for group_key in shuffled_group_keys:
            if group_key != max_id:
                # 마지막 그룹은 제외하고 섞기
                group_indices = self.groups[group_key]
                # Shuffle within each group
                #random.shuffle(group_indices)
                all_indices.extend(group_indices)
        
        # 마지막 그룹은 맨 뒤에 추가
        group_indices = self.groups[max_id]
        random.shuffle(group_indices)
        all_indices.extend(group_indices)
        
        return all_indices
    
    def __iter__(self):
        # Shuffle groups themselves
        all_indices = self._get_curriculum_indices()
        
        # Create batches from the shuffled indices
        # This part requires careful implementation to ensure batch_size respected
        # and potentially handling uneven group sizes
        cur_start_idx = 0
        while cur_start_idx < len(all_indices):
            cur_batch = all_indices[cur_start_idx:cur_start_idx + self.batch_size]
            cur_start_idx += self.batch_size
            
            yield cur_batch

    def __len__(self):
        # Calculate the total number of batches
        return self.batch_num

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def set_contrastive_parameters(self, 
            temperature_parameter: float = 1.0,
            generation_lambda: float = 1.0,
            contrastive_lambda: float = 1.0,
        ):
        self.temperature_parameter = temperature_parameter
        self.generation_lambda = generation_lambda
        self.contrastive_lambda = contrastive_lambda
    
    def set_training_method(self, train_method: str = "standard"):
        self.train_method = train_method
    
    def compute_loss(self, model, 
                    inputs: dict[str, Union[torch.Tensor, Any]],
                    return_outputs: bool = False,
                    num_items_in_batch: Optional[torch.Tensor] = None,
        ) -> Union[torch.Tensor, tuple[torch.Tensor, Any]]:
        outputs = model(**inputs)
        contrastive_label = inputs.get("contrastive_label")
        decoder_label = inputs.get("labels")
        generation_loss = outputs.get("loss")
        encoder_last_hidden = outputs.get("encoder_last_hidden_state")
        decoder_last_hidden = outputs.get("decoder_hidden_states")
        
        loss = generation_loss
        if self.train_method == "contrastive" and contrastive_label is not None:
            # contrastive loss 계산
            # batch wide로 loop를 돌면서 hidden state에서 padding된 부분을 제거하여 pooling 수행
            batch_size = encoder_last_hidden.size(0)
            encoder_hidden_pooled = torch.zeros((batch_size, encoder_last_hidden.size(2)), device=encoder_last_hidden.device)
            decoder_hidden_pooled = torch.zeros((batch_size, decoder_last_hidden.size(2)), device=decoder_last_hidden.device)
            for batch_idx in range(batch_size):
                cur_batch_encoder_mask = inputs['attention_mask'][batch_idx].bool()  # (seq_len, )
                cur_batch_decoder_mask = decoder_label[batch_idx].ne(-100)  # (tgt_seq_len, )
                
                # mixed pooling
                encoder_hidden_pooled[batch_idx] = torch.sum(encoder_last_hidden[batch_idx][cur_batch_encoder_mask], dim=0) + torch.mean(encoder_last_hidden[batch_idx][cur_batch_encoder_mask], dim=0)
                decoder_hidden_pooled[batch_idx] = torch.sum(decoder_last_hidden[batch_idx][cur_batch_decoder_mask], dim=0) + torch.mean(decoder_last_hidden[batch_idx][cur_batch_decoder_mask], dim=0)
                
            
            # InfoNCE loss
            # simmilarity 척도는 cosine 유사도
            cosine_sim = torch.nn.functional.cosine_similarity(encoder_hidden_pooled.unsqueeze(1), decoder_hidden_pooled.unsqueeze(0), dim=2)  # (batch_size, batch_size)
            
            positive_mask = torch.ones(cosine_sim.size(0), device=cosine_sim.device).bool()
            if contrastive_label is not None:
                positive_mask = contrastive_label.bool()
            
            positive_mask.requires_grad = False
            
            positive_values = torch.diagonal(cosine_sim)[positive_mask]  # (num_positives, )
            positive_silimarities = torch.exp(positive_values / self.temperature_parameter)  # (batch_size,)
            
            # 분모 계산 전에 자기 샘플에 해당하는 negative 샘플만 더하도록 마스크 생성
            '''start_idx = 0
            negative_sample_mask = list()
            for i in range(1, contrastive_label.size(0)):
                if contrastive_label[i] == 1:
                    cur_mask = [0] * (start_idx) + [1] * (i - start_idx) + [0] * (cosine_sim.size(0) - i)
                    negative_sample_mask.append(cur_mask)
                    start_idx = i
            # 마지막 샘플 처리
            if len(negative_sample_mask) < positive_silimarities.size(0):
                cur_mask = [0] * (start_idx) + [1] * (cosine_sim.size(0) - start_idx)
                negative_sample_mask.append(cur_mask)
            
            negative_sample_mask = torch.tensor(negative_sample_mask, device=cosine_sim.device, requires_grad=False).int() # (num_positives, batch_size)'''
            # 여기까지 마스크 생성
            # 마스크 적용시 성능 하락 발생
            
            exp_simmilarities = torch.exp(cosine_sim / self.temperature_parameter)[positive_mask] # (num_positives, batch_size)
            #exp_simmilarities = exp_simmilarities * negative_sample_mask  # (num_positives, batch_size)
            exp_simmilarities = torch.sum(exp_simmilarities, dim=1)  # (num_positives, )
            
            contrastive_loss = -torch.log(positive_silimarities / exp_simmilarities).mean()

            # 두 loss 합치기
            loss = self.generation_lambda * generation_loss + self.contrastive_lambda * contrastive_loss
        
        return (loss, outputs) if return_outputs else loss
    
    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (Union[`Dataset`, dict[str, `Dataset`]), *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. If it is a dictionary, it will
                evaluate on each dataset, prepending the dictionary key to the metric name. Datasets must implement the
                `__len__` method.

                <Tip>

                If you pass a dictionary with names of datasets as keys and datasets as values, evaluate will run
                separate evaluations on each dataset. This can be useful to monitor how training affects other
                datasets or simply to get a more fine-grained evaluation.
                When used with `load_best_model_at_end`, make sure `metric_for_best_model` references exactly one
                of the datasets. If you, for example, pass in `{"data1": data1, "data2": data2}` for two datasets
                `data1` and `data2`, you could specify `metric_for_best_model="eval_data1_loss"` for using the
                loss on `data1` and `metric_for_best_model="eval_data2_loss"` for the loss on `data2`.

                </Tip>

            ignore_keys (`list[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        
        # for seq2seq trainer
        gen_kwargs = gen_kwargs.copy()

        # Use legacy argument setting if a) the option is not explicitly passed; and b) the argument is set in the
        # training args
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
            and self.args.generation_max_length is not None
        ):
            gen_kwargs["max_length"] = self.args.generation_max_length
        if gen_kwargs.get("num_beams") is None and self.args.generation_num_beams is not None:
            gen_kwargs["num_beams"] = self.args.generation_num_beams
        # We don't want to drop samples in general
        self.gather_function = self.accelerator.gather
        self._gen_kwargs = gen_kwargs
        
        # handle multiple eval datasets
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset if override else eval_dataset_name,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        if f"{metric_key_prefix}_model_preparation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_model_preparation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics
    
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                or (self.is_fsdp_enabled and self.accelerator.mixed_precision != "fp8" and not self.args.torch_compile)
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"\n***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        if hasattr(model, "eval") and callable(model.eval):
            model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        all_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_preds = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_labels = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_inputs = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)

        metrics = None
        eval_set_kwargs = {}

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = (
                self._prepare_input(inputs[main_input_name]) if "inputs" in args.include_for_metrics else None
            )

            # Update containers
            if losses is not None:
                losses = self.gather_function(losses.repeat(batch_size))
                all_losses.add(losses)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function(inputs_decode)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_inputs.add(inputs_decode)
            if labels is not None:
                # Pad labels here, preparing for preprocess_logits_for_metrics in next logits block.
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function(logits)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_preds.add(logits)
            if labels is not None:
                labels = self.gather_function(labels)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels.add(labels)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            if self.args.batch_eval_metrics:
                if self.compute_metrics is not None and logits is not None and labels is not None:
                    is_last_step = self.accelerator.gradient_state.end_of_dataloader
                    batch_kwargs = {}
                    batch_kwargs["losses"] = losses if "loss" in args.include_for_metrics else None
                    batch_kwargs["inputs"] = inputs if "inputs" in args.include_for_metrics else None
                    metrics = self.compute_metrics(
                        EvalPrediction(predictions=logits, label_ids=labels, **batch_kwargs),
                        compute_result=is_last_step,
                    )

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            elif args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                all_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                all_inputs.to_cpu_and_numpy()

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        all_losses = all_losses.get_arrays()
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()
        all_inputs = all_inputs.get_arrays()

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if (
            self.compute_metrics is not None
            and all_preds is not None
            and all_labels is not None
            and not self.args.batch_eval_metrics
        ):
            eval_set_kwargs["losses"] = all_losses if "loss" in args.include_for_metrics else None
            eval_set_kwargs["inputs"] = all_inputs if "inputs" in args.include_for_metrics else None
            metrics = self.compute_metrics(
                EvalPrediction(predictions=all_preds, label_ids=all_labels, **eval_set_kwargs)
            )
        elif metrics is None:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if isinstance(all_losses, list) and all_losses:
            metrics[f"{metric_key_prefix}_loss"] = np.concatenate(all_losses).mean().item()
        elif isinstance(all_losses, np.ndarray):
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time
        if hasattr(self, "model_preparation_time"):
            metrics[f"{metric_key_prefix}_model_preparation_time"] = self.model_preparation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`list[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = False if len(self.label_names) == 0 else any(inputs.get(k) is not None for k in self.label_names) # 내가 진행하는 실험에서는 굳이 모든 label_names가 input에 포함될 필요가 없다.
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss")
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = len(self.label_names) == 0 and return_loss

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", ["past_key_values"])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        num_items_in_batch = self._get_num_items_in_batch([inputs], self.args.device)
                        loss, outputs = self.compute_loss(
                            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
                        )
                    loss = loss.detach().mean()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)
    
    def get_train_dataloader(self):
        # Assuming self.train_dataset has a 'group_id' column
        group_ids = []
        index = 0
        for example in self.train_dataset:
            if 'group_id' in example:
                group_ids.append(example['group_id'])
            else:
                group_ids.append(index)
            index += 1
        
        if self.train_method == "contrastive":
            sampler = ContrastiveGroupedBatchSampler(self.train_dataset, group_ids, self.args.per_device_train_batch_size)
        elif self.train_method == "curriculum":
            sampler = CurriculumGroupedBatchSampler(self.train_dataset, group_ids, self.args.per_device_train_batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(self.train_dataset)
        
        # 이제 쓸모 없어진 group_id 컬럼을 제거합니다.
        if 'group_id' in self.train_dataset.column_names:
            self.train_dataset = self.train_dataset.remove_columns(['group_id'])
        
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        # Assuming self.train_dataset has a 'group_id' column
        group_ids = []
        index = 0
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        for example in eval_dataset:
            if 'group_id' in example:
                group_ids.append(example['group_id'])
            else:
                group_ids.append(index)
            index += 1
        
        if self.train_method == "contrastive":
            sampler = ContrastiveGroupedBatchSampler(eval_dataset, group_ids, self.args.per_device_eval_batch_size)
        elif self.train_method == "curriculum":
            sampler = CurriculumGroupedBatchSampler(eval_dataset, group_ids, self.args.per_device_eval_batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(eval_dataset)
        
        # 이제 쓸모 없어진 group_id 컬럼을 제거합니다.
        if 'group_id' in eval_dataset.column_names:
            eval_dataset = eval_dataset.remove_columns(['group_id'])
        
        return torch.utils.data.DataLoader(
            eval_dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )