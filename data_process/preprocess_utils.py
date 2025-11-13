import json
import pandas as pd
import pathlib

def save_entity_types(entity_file_path, dataset_name, entity_types):
    entity_file = pathlib.Path(entity_file_path)
    entity_file.parent.mkdir(parents=True, exist_ok=True)
    
    if entity_file.exists():
        print(f"{entity_file_path} already exists. Loading existing entity types.")
        existing_types = None
        with open(entity_file_path, 'r', encoding='utf-8') as f:
            existing_types = json.loads(f.read())
        
        existing_types[dataset_name] = entity_types
        with open(entity_file_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(existing_types, ensure_ascii=False))
    else:
        print(f"{entity_file_path} does not exist. Creating new entity types file.")
        with open(entity_file_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps({dataset_name: entity_types}, ensure_ascii=False))

def split_train_dev(train_csv_file_path, dev_csv_file_path, dev_ratio=0.1):
    df = pd.read_csv(train_csv_file_path)
    dev_size = int(len(df) * dev_ratio)
    dev_df = df.sample(n=dev_size, random_state=42)
    train_df = df.drop(dev_df.index)

    train_df.to_csv(train_csv_file_path, index=False)
    dev_df.to_csv(dev_csv_file_path, index=False)

def mix_big_and_small(data_directory, split, ratio=0.5):
    big_df = pd.read_csv(f"{data_directory}/few-nerd/supervised/{split}.preprocessed.big.csv")
    small_df = pd.read_csv(f"{data_directory}/few-nerd/supervised/{split}.preprocessed.small.csv")

    # Sample from the small dataframe
    small_sample = small_df.sample(frac=ratio)
    small_sample.loc[:, "types"] = "fewnerd_small"
    
    small_index = small_sample.index
    big_sample = big_df[~big_df.index.isin(small_index)]
    big_sample.loc[:, "types"] = "fewnerd_big"

    combined_df = pd.concat([big_sample, small_sample], ignore_index=True)
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)  # Shuffle the combined dataframe
    return combined_df

def add_types_column_to_dataset(data_file_path, dataset_name):
    df = pd.read_csv(data_file_path)
    df.loc[:, "types"] = dataset_name
    return df

# entity list를 데이터셋 파일에서 제시하고, json은 인덱스로 key를 생성하도록 데이터셋을 변경
# 인덱스 리스트는 무작위로 섞음, 길이도 무작위로 자름
# 길이는 0.5~1.0 비율로 자름
def use_entity_list_for_entity_types_index(data_file_path, entity_types_dict_path, min_length_ratio=0.5):
    import random
    import copy
    entity_lists = []
    df = pd.read_csv(data_file_path)
    
    with open(entity_types_dict_path, 'r', encoding='utf-8') as f:
        entity_types_dict = json.loads(f.read())

    for idx, row in df.iterrows():
        dataset_name = row['types']
        entity_list = copy.deepcopy(entity_types_dict.get(dataset_name, []))
        cur_json_dict = json.loads(row["NER"])
        
        # entity list에서 필요한 key의 개수를 구하기
        cur_type_num = len(cur_json_dict)
        
        # 길이의 비율을 추출한다.
        length_ratio = random.uniform(min_length_ratio, 1.0)
        sampled_entity_num = int(len(entity_list) * length_ratio)
        
        # entity 전체 list에서 현재 key를 제외한다.
        for k in cur_json_dict.keys():
            if k in entity_list:
                entity_list.remove(k)
        
        # 모자란 type 개수만큼 entity list에서 추출
        deficit = sampled_entity_num - cur_type_num
        if deficit > 0:
            sampled_entities = random.sample(entity_list, min(deficit, len(entity_list)))
            entity_list = sampled_entities
        
        entity_list.extend(list(cur_json_dict.keys()))

        # shuffle the entity list
        random.shuffle(entity_list)
        
        # Json을 인덱스 key 형태로 변경
        json_dict_indexed = dict()
        for k, v in cur_json_dict.items():
            if k in entity_list:
                json_dict_indexed[entity_list.index(k)] = v
            else:
                raise ValueError(f"Entity type {k} not found in entity list.")

        entity_lists.append(" ".join(entity_list))
        df.at[idx, "NER"] = json.dumps(json_dict_indexed, ensure_ascii=False)

    df.loc[df.index, "entity_list"] = entity_lists
    return df

# key를 인덱스가 아닌 entity type 문자열로 유지
# entity list는 무작위로 섞음, 길이도 무작위로 자름
# 이때 정답 entity type이 포함되지 않을 수 있으며, 그럴 때는 json에서 해당 key를 예측하지 않도록 함
def use_entity_list_for_entity_types_full(data_file_path, entity_types_dict_path, min_length_ratio=0.5, max_length_ratio=1.0):
    import random
    import copy
    entity_lists = []
    df = pd.read_csv(data_file_path)
    
    with open(entity_types_dict_path, 'r', encoding='utf-8') as f:
        entity_types_dict = json.loads(f.read())

    for idx, row in df.iterrows():
        dataset_name = row['types']
        entity_list = copy.deepcopy(entity_types_dict.get(dataset_name, []))
        cur_json_dict = json.loads(row["NER"])
        
        # 길이의 비율을 추출한다.
        length_ratio = random.uniform(min_length_ratio, max_length_ratio)
        sampled_entity_num = int(len(entity_list) * length_ratio)
        
        
        # type 개수만큼 entity list에서 추출
        sampled_entities = random.sample(entity_list, sampled_entity_num)
        entity_list = sampled_entities

        # shuffle the entity list
        random.shuffle(entity_list)
        
        # Json을 인덱스 key 형태로 변경
        json_dict_indexed = dict()
        for k, v in cur_json_dict.items():
            if k in entity_list:
                json_dict_indexed[k] = v

        entity_lists.append(" ".join(entity_list))
        df.at[idx, "NER"] = json.dumps(json_dict_indexed, ensure_ascii=False)

    df.loc[df.index, "entity_list"] = entity_lists
    return df

# dictionary 형태 NER Json 객체를 받아서 Json과는 다른 단순한 형태의 문자열로 변환
def dict_to_structured_string(json_dict):
    structured_string = ""
    for key, value in json_dict.items():
        structured_string += f"{key}: {'  '.join(value)}\n"
    return structured_string

def remove_brackets_from_ner_string(ner_string):
    ner_string = ner_string.replace("{", "").replace("}", "")

    return ner_string

def convert_ner_json_to_simplified_format(data_file_path, output_file_path):
    df = pd.read_csv(data_file_path)
    simplified_ner_list = []
    
    for idx, row in df.iterrows():
        simplified_string = remove_brackets_from_ner_string(row["NER"])
        simplified_ner_list.append(simplified_string.strip())
    
    df.loc[df.index, "NER"] = simplified_ner_list
    df.to_csv(output_file_path, index=False)