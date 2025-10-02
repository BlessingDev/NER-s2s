# 금방 하지
# 데이터셋에서 NER인 부분만 binary로 labeling하는 코드
import os
import json
import pandas as pd

def proprocess_ner_to_binary(file_path):
    """
    Preprocesses FewNERD CoNLL data from a file and returns a DataFrame.

    Args:
        file_path (str): Path to the MIT CoNLL formatted file.
        
    Returns:
        pd.DataFrame: DataFrame containing the preprocessed data.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    data_list = []
    cur_sentence = []
    ner_list = []
    cur_ner = None
    for line in lines:
        if line == '\n': # 문장 경계
            if len(cur_sentence) > 0:
                
                data_list.append({
                    "Sentence": " ".join(cur_sentence).strip(),
                    "NER": " ".join(ner_list).strip()
                })
                cur_sentence = []
                ner_list = []
        else:
            # 각 줄에서 단어, 품사, 구, NER 정보를 추출
            word, ner = line.strip().split()
            
            cur_sentence.append(word)
            
            # NER 정보 처리
            if ner != "O":
                ner_list.append("1")
            else:
                ner_list.append("0")
        
    # 마지막 문장 저장
    if len(cur_sentence) > 0:
        data_list.append({
            "Sentence": " ".join(cur_sentence).strip(),
            "NER": " ".join(ner_list).strip()
        })

    df = pd.DataFrame(data_list)
    return df

def preprocess_binary_to_tokenization(word_binary_df, tokenizer):
    """
    Preprocesses the DataFrame to align NER labels with tokenized inputs.

    Args:
        df (pd.DataFrame): DataFrame containing sentences and NER labels.
        
    Returns:
        pd.DataFrame: DataFrame with tokenized sentences and aligned labels.
    """
    data_list = []
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
        
        # word_id를 str로 변환
        str_word_ids = list(map(str, whole_encoded["input_ids"]))
        # label에 "\s" 토큰 한개 추가로 넣어주기
        aligned_labels.append("0")

        data_list.append({
            "Sentence": " ".join(str_word_ids),
            "Label": " ".join(aligned_labels)
        })
    
    df = pd.DataFrame(data_list)
        
    return df