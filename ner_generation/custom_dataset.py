import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
import pandas as pd
from tqdm.auto import tqdm
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
from transformers import DataCollatorForSeq2Seq

from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

@dataclass
class DataCollatorForSeq2SeqAndEncoderTokenClassification(DataCollatorForSeq2Seq):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`], *optional*):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.0 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        label_name = "label" if "label" in features[0] else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0] else None
        # reconvert list[None] to None if necessary
        # this might occur when we pass {..., "labels": None}
        if labels is not None and all(label is None for label in labels):
            labels = None
        non_labels_features = [{k: v for k, v in feature.items() if label_name not in k} for feature in features]

        # run through tokenizer without labels to ensure no side effects
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        
        # encoder_labels도 패딩해주기
        encoder_labels = [feature["encoder_labels"] for feature in features] if "encoder_labels" in features[0] else None
        if encoder_labels is not None:
            max_label_length = max(len(l) for l in encoder_labels)
            padding_side = self.tokenizer.padding_side
            if isinstance(encoder_labels[0], list):  
                # classification은 -100으로 패딩
                batch["encoder_labels"] = [
                    label + [-100] * (max_label_length - len(label))
                    if padding_side == "right"
                    else [-100] * (max_label_length - len(label)) + label
                    for label in encoder_labels
                ]

        # we have to pad the labels manually as we cannot rely on `tokenizer.pad` and we need them to be of the same length to return tensors
        no_padding = self.padding is False or self.padding == PaddingStrategy.DO_NOT_PAD
        if labels is not None:
            if no_padding:
                if isinstance(features[0][label_name], list):
                    batch["labels"] = list(labels)
                else:
                    batch["labels"] = [np.concatenate([label, []]) for label in labels]
            else:
                max_padding = self.padding == PaddingStrategy.MAX_LENGTH and self.max_length is not None
                max_label_length = max(len(l) for l in labels) if not max_padding else self.max_length
                if self.pad_to_multiple_of is not None:
                    max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                    )

                padding_side = self.tokenizer.padding_side
                if isinstance(features[0][label_name], list):
                    batch["labels"] = [
                        label + [self.label_pad_token_id] * (max_label_length - len(label))
                        if padding_side == "right"
                        else [self.label_pad_token_id] * (max_label_length - len(label)) + label
                        for label in labels
                    ]
                else:
                    batch["labels"] = [
                        np.concatenate(
                            [
                                label,
                                np.array([self.label_pad_token_id] * (max_label_length - len(label)), dtype=np.int64),
                            ]
                        )
                        if padding_side == "right"
                        else np.concatenate(
                            [
                                np.array([self.label_pad_token_id] * (max_label_length - len(label)), dtype=np.int64),
                                label,
                            ]
                        )
                        for label in labels
                    ]

        # reintroduce side effects via tokenizer that return respective datatypes for the `return_tensors` argument
        if batch.get("labels", None) is not None:
            if return_tensors == "pt":
                import torch

                batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)
                
                # 내 사전에 tensorflow는 없다
            else:
                batch["labels"] = np.array(batch["labels"], dtype=np.int64)
        else:
            batch["labels"] = None
            
        if batch.get("encoder_labels", None) is not None:
            if return_tensors == "pt":
                import torch

                batch["encoder_labels"] = torch.tensor(batch["encoder_labels"], dtype=torch.int64)
                
                # 내 사전에 tensorflow는 없다
            else:
                batch["encoder_labels"] = np.array(batch["encoder_labels"], dtype=np.int64)
        else:
            batch["encoder_labels"] = None

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=batch["labels"])
            batch["decoder_input_ids"] = decoder_input_ids

        return batch

class TokenClassificationDataset(Dataset):
    """
    Custom PyTorch Dataset for token labeling tasks.
    It takes raw CSV-like text data, tokenizes it, and aligns the word-level 
    labels to the tokenizer's subword outputs.
    """
    def __init__(self, classification_data_path: str, tokenizer, label_names):
        """
        Args:
            data_path (str): Path to the CSV file containing the dataset.
        """
        self.tokenizer = tokenizer
        
        if len(label_names) == 1:
            # binary classification일 때 label names를 0, 1로 설정
            label_names = [0, 1]
        
        self.label_names = label_names

        tc_dataset = HFDataset.from_csv(classification_data_path)
        tc_encoded_dataset = tc_dataset.map(self.preprocess_binary_to_tokenization, batched=True)

        self.sentences = tc_encoded_dataset["sentence_encoded"]
        self.labels = tc_encoded_dataset["label_encoded"]

    def preprocess_binary_to_tokenization(self, batch_samples):
        """
        Preprocesses the DataFrame to align NER labels with tokenized inputs.

        Args:
            df (pd.DataFrame): DataFrame containing sentences and NER labels.
            
        Returns:
            pd.DataFrame: DataFrame with tokenized sentences and aligned labels.
        """
        sentence_list = []
        labels_list = []
        length_list = []
        name_to_index = {str(name): idx for idx, name in enumerate(self.label_names)}
        
        for idx in range(len(batch_samples["Sentence"])):
            sentence = batch_samples["Sentence"][idx]
            ner_labels = batch_samples["NER"][idx].split()

            word_list = sentence.split()
            
            aligned_labels = list()
            
            word_progress_list = [
                ' '.join(word_list[:idx+1]) for idx in range(len(word_list))
            ]
            
            encodings = self.tokenizer(
                word_progress_list, add_special_tokens=False
            )
            
            alerted = False
            for word_idx in range(len(word_list)):
                # Tokenize the sentence
                encoding = encodings["input_ids"][word_idx]
                
                if not alerted and self.tokenizer.unk_token_id in encoding:
                    #print(f"Unknown token found in sentence: {sentence}, word: {word_list[word_idx]}")
                    alerted = True
                
                cur_length = len(encoding) # exclude '<\s>' token
                cur_label = ner_labels[word_idx]
                
                cur_label_idx = name_to_index[cur_label] if cur_label != "-100" else -100
                
                added_length = cur_length - len(aligned_labels)

                aligned_labels.extend([cur_label_idx] * added_length)

            whole_encoded = encodings["input_ids"][-1]
            
            # 인코더 디코더 동시 학습에서는 필요 없음
            length_list.append(len(whole_encoded))

            assert len(whole_encoded) == len(aligned_labels), f"Length mismatch: {len(whole_encoded)} vs {len(aligned_labels)}"
            
            sentence_list.append(whole_encoded)
            labels_list.append(aligned_labels)
            
        
        #print("Max length:", max(length_list))
        
        return {
            "sentence_encoded": sentence_list,
            "label_encoded": labels_list
        }
    
    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.sentences)

    def __getitem__(self, index):
        """
        Fetches a sample and prepares it for the model.

        This involves tokenizing the sentence and aligning the labels with the
        generated subword tokens.
        """

        torch.cuda.empty_cache()
        item = dict()
        
        if isinstance(index, slice):
            item = []
            for i in range(*index.indices(len(self))):
                cur_item = dict()
                cur_item['input_ids'] = torch.as_tensor(self.sentences[i])
                cur_item['labels'] = torch.as_tensor(self.labels[i])
                item.append(cur_item)
        else:
            sentence = self.sentences[index]
            item['input_ids'] = torch.as_tensor(sentence)
            
            word_labels = self.labels[index]
            item['labels'] = torch.as_tensor(word_labels)

        return item

class TokenClassificationTestDataset(TokenClassificationDataset):
    """
    Custom PyTorch Dataset for token labeling tasks.
    It takes raw CSV-like text data, tokenizes it, and aligns the word-level 
    labels to the tokenizer's subword outputs.
    """
    def __init__(self, classification_data_path: str, tokenizer, label_names):
        """
        Args:
            data_path (str): Path to the CSV file containing the dataset.
        """
        super().__init__(classification_data_path, tokenizer, label_names)

    
    def __getitem__(self, index):
        """
        Fetches a sample and prepares it for the model.

        This involves tokenizing the sentence and aligning the labels with the
        generated subword tokens.
        """

        torch.cuda.empty_cache()
        item = dict()
        
        if isinstance(index, slice):
            item = []
            for i in range(*index.indices(len(self))):
                cur_item = dict()
                cur_item['input_ids'] = torch.as_tensor(self.sentences[i])
                item.append(cur_item)
        else:
            sentence = self.sentences[index]
            item['input_ids'] = torch.as_tensor(sentence)

        return item
    
    def get_label_ratio(self):
        """
        Returns the ratio of positive labels in the dataset.
        """
        total_count = 0
        positive_count = 0
        
        for labels in tqdm(self.labels, desc="Calculating label ratio"):
            for label in labels:
                if label != -100:
                    total_count += 1
                    if label != 0:
                        positive_count += 1
        
        ratio = positive_count / total_count if total_count > 0 else 0
        return ratio

class FewNerdGenerationDataset(Dataset):
    """
    Custom PyTorch Dataset for generation tasks.
    It takes raw CSV-like text data and prepares it for the model.
    """
    def __init__(self, data_path: str):
        """
        Args:
            data_path (str): Path to the CSV file containing the dataset.
        """
        self.data = pd.read_csv(data_path)

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, index):
        """Fetches a sample from the dataset."""
        item = self.data.iloc[index]
        return {
            'prompts': torch.tensor(item['sentence'], dtype=torch.long),
        }