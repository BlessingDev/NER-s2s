import torch
from torch.utils.data import Dataset
import pandas as pd
import csv
import io

def preprocess_binary_to_tokenization(word_binary_df, tokenizer):
    """
    Preprocesses the DataFrame to align NER labels with tokenized inputs.

    Args:
        df (pd.DataFrame): DataFrame containing sentences and NER labels.
        
    Returns:
        pd.DataFrame: DataFrame with tokenized sentences and aligned labels.
    """
    data_list = []
    length_list = []
    
    for idx, row in word_binary_df.iterrows():
        sentence = row['Sentence']
        ner_labels = row['NER'].split()
        
        word_list = sentence.split()
        
        aligned_labels = list()
        
        for word_idx in range(len(word_list)):
            # Tokenize the sentence
            encoding = tokenizer(
                word_list[:word_idx+1],  # Split sentence into words
                is_split_into_words=True,
                return_offsets_mapping=False,
                truncation=True,
            )
            
            cur_length = len(encoding['input_ids']) - 1 # exclude '<\s>' token
            cur_label = ner_labels[word_idx]
            
            added_length = cur_length - len(aligned_labels)
            
            aligned_labels.extend([cur_label] * added_length)
        
        whole_encoded = tokenizer(
            word_list,  # Split sentence into words
            is_split_into_words=True,
            return_offsets_mapping=False,
            truncation=True,
        )
        
        # label에 "\s" 토큰 한개 추가로 넣어주기
        aligned_labels.append(0)
        length_list.append(len(whole_encoded['input_ids']))
        
        data_list.append({
            "Sentence": whole_encoded["input_ids"],
            "Label": aligned_labels
        })
    
    print("Max length:", max(length_list))
    
    df = pd.DataFrame(data_list)
    
    return df

class FewNerdBinaryDataset(Dataset):
    """
    Custom PyTorch Dataset for token labeling tasks.
    It takes raw CSV-like text data, tokenizes it, and aligns the word-level 
    labels to the tokenizer's subword outputs.
    """
    def __init__(self, binary_data_path: str, tokenizer):
        """
        Args:
            data_path (str): Path to the CSV file containing the dataset.
        """
        self.tokenizer = tokenizer

        binary_df = pd.read_csv(binary_data_path)

        print("data processing start")
        processed_df = preprocess_binary_to_tokenization(binary_df, self.tokenizer)
        print("data processing finished...")

        self.sentences = []
        self.labels = []

        # Read and process each row
        for _, row in processed_df.iterrows():
            cur_sentence = [int(token_idx) for token_idx in row['Sentence']]
            # Split the NER string and convert labels to integers
            cur_labels = [int(label) for label in row["Label"]]
            
            self.sentences.append(cur_sentence)
            self.labels.append(cur_labels)

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.sentences)

    def __getitem__(self, index):
        """
        Fetches a sample and prepares it for the model.

        This involves tokenizing the sentence and aligning the labels with the
        generated subword tokens.
        """
        sentence = self.sentences[index]
        word_labels = self.labels[index]

        item = dict()
        
        if isinstance(index, slice):
            item = []
            for i in range(*index.indices(len(self))):
                cur_item = dict()
                cur_item['input_ids'] = torch.as_tensor(self.sentences[i])
                cur_item['labels'] = torch.as_tensor(self.labels[i])
                item.append(cur_item)
        else:
            item['input_ids'] = torch.as_tensor(sentence)
            item['labels'] = torch.as_tensor(word_labels)

        return item

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