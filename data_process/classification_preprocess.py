# 금방 하지
# 데이터셋에서 NER인 부분만 binary로 labeling하는 코드
import os
import json
import pandas as pd

def proprocess_fewnerd_to_binary(file_path):
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

def preprocess_conll_to_binary(file_path):
    """
    Preprocesses FewNERD CoNLL data from a file and returns a DataFrame.

    Args:
        file_path (str): Path to the MIT CoNLL formatted file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
    data_list = []
    cur_sentence = []
    ner_list = []
    cur_ner = None
    for line in lines[1:]: # 헤더 스킵
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
            word, pos, chunk, ner = line.strip().split()
            
            ner = ner.split("-")[-1]  # B-PER -> PER
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

def preprocess_sim_conll_to_classification(file_path, is_bio=False):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    data_list = []
    cur_sentence = []
    ner_list = []
    cur_ner = None
    for line in lines: # 헤더 스킵
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
            
            if is_bio and ner != "O":
                ner = '-'.join(ner.split("-")[1:])
            
            cur_sentence.append(word)
            
            # NER 정보 처리
            if ner != "O":
                ner_list.append(ner)
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

def preprocess_conll_to_classification(file_path):
    """
    Preprocesses CoNLL2003 data from a file and returns a DataFrame.

    Args:
        file_path (str): Path to the CoNLL formatted file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    str_to_index = dict()
    data_list = []
    cur_sentence = []
    ner_list = []
    cur_ner = None
    for line in lines[1:]: # 헤더 스킵
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
            word, pos, chunk, ner = line.strip().split()
            
            ner = ner.split("-")[-1]  # B-PER -> PER
            cur_sentence.append(word)
            
            # NER 정보 처리
            if ner != "O":
                ner_list.append(ner)
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

def mit_ner_to_classification(file_path):
    """
    Preprocesses MIT NER data from a file and returns a DataFrame.

    Args:
        file_path (str): Path to the MIT CoNLL formatted file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    data_list = []
    cur_sentence = []
    ner_list = []
    for line in lines[1:]: # 헤더 스킵
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
            ner, word = line.strip().split()
            
            ner = ner.split("-")[-1]  # B-PER -> PER
            cur_sentence.append(word)
            
            # NER 정보 처리
            if ner != "O":
                ner_list.append(ner)
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

def preprocess_classification_to_switch(word_classification_df_path, dataset_name, type_ratio=0.0, line_identifier="//", simple_prompt=False):
    """
    Preprocesses the DataFrame to learn switching tokens for classification.
    """
    if simple_prompt:
        prompts_template = "{line} {entity} {line} "
    else:
        prompts_template = "Determine whether or not the named entity of type {line} {entity} {line} is present in the following sentence {line} "
    entity_types = list()
    with open("/workspace/datas/entity_types.json", "r", encoding="utf-8") as f:
        json_data = json.loads(f.read())
        entity_types = json_data[dataset_name]

    word_classification_df = pd.read_csv(word_classification_df_path)
    prompts = []
    switch_tags = []
    for index, row in word_classification_df.iterrows():
        
        sentence = row['Sentence']
        ner_labels = row['NER'].split()
        
        ner_set = set(ner_labels)
        ner_set.discard("0")  # non-entity 제거
        
        cur_entity_types = list(ner_set)
        
        if type_ratio > 0.0:
            import random
            
            cur_pool = list(set(entity_types) - set(cur_entity_types))
            select_num = max(1, int(len(entity_types) * type_ratio))
            defect_number = select_num - len(cur_entity_types)
            if defect_number > 0:
                if dataset_name == "fewnerd_small":
                    # fewnerd small에서는 비슷한 유형끼리 negative 샘플링하도록 함
                    # 먼저 현재 entity type의 상위 유형을 찾음
                    big_type_set = set([ner.split("-")[0] for ner in ner_set])
                    # building과 location은 친연관계
                    if "building" in big_type_set or "location" in big_type_set:
                        big_type_set.add("building")
                        big_type_set.add("location")
                    
                    cur_pool = [ner for ner in cur_pool if ner.split("-")[0] in big_type_set]
                    defect_number = min(defect_number, len(cur_pool))
                cur_entity_types = random.sample(cur_pool, defect_number)
                cur_entity_types.extend(list(ner_set))
        
        for ner in cur_entity_types:
            cur_prompts = prompts_template.format(entity=ner, line=line_identifier)
            prompt_len = len(cur_prompts.split())
            cur_switch_tags = ["-100"] * prompt_len
            for label in ner_labels:
                if label == ner:
                    cur_switch_tags.append("1")
                else:
                    cur_switch_tags.append("0")

            prompts.append(cur_prompts + sentence)
            switch_tags.append(' '.join(cur_switch_tags))

    df = pd.DataFrame({
        "Sentence": prompts,
        "NER": (switch_tags)
    })

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