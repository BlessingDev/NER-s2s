import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
import pandas as pd
from tqdm.auto import tqdm

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
        tc_encoded_dataset = tc_dataset.map(self.preprocess_binary_to_tokenization, batched=True, batch_size=100)

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
                word_progress_list
            )
            
            for word_idx in range(len(word_list)):
                # Tokenize the sentence
                encoding = encodings["input_ids"][word_idx]
                
                cur_length = len(encoding) - 1 # exclude '<\s>' token
                cur_label = ner_labels[word_idx]
                
                cur_label_idx = name_to_index[cur_label] if cur_label != "-100" else -100
                
                added_length = cur_length - len(aligned_labels)

                aligned_labels.extend([cur_label_idx] * added_length)

            whole_encoded = encodings["input_ids"][-1]
            
            # label에 "\s" 토큰 한개 추가로 넣어주기
            aligned_labels.append(0)
            length_list.append(len(whole_encoded))

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