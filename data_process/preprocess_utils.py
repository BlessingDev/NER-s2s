import json
import pandas as pd
import pathlib
from tqdm.auto import tqdm

def inerd_to_tag_list(word_list, type_list, sentence):
    sentence_words = sentence.split()
    tag_list = list()
    
    if len(word_list) == 0:
        return tag_list
    
    word_idx = 0
    
    for idx, sentence_word in enumerate(sentence_words):
        cur_start_word = word_list[word_idx].split()[0]
        while cur_start_word not in sentence_words[idx:]:
            word_idx += 1
            if word_idx >= len(word_list):
                break
            cur_start_word = word_list[word_idx].split()[0]
        
        while sentence_word == cur_start_word:
            cur_tag_word_len = len(word_list[word_idx].split())
            sentence_match_span = ' '.join(sentence_words[idx:idx + cur_tag_word_len])
            if sentence_match_span == word_list[word_idx]:
                # 매칭 성공
                tag_list.append((idx, idx + cur_tag_word_len, type_list[word_idx]))
                
                word_idx += 1
                if word_idx >= len(word_list):
                    break
                
                cur_start_word = word_list[word_idx].split()[0]
            else:
                break
        
        if word_idx >= len(word_list):
            break
    
    return tag_list

def inerd_to_json(inerd_str, inerd_version=1):
    json_obj = dict()
    if len(inerd_str.strip()) > 0:
        entity_list = inerd_str.split("<ES>")
        for entity_str in entity_list:
            if len(entity_str.strip()) == 0:
                continue
            try:
                entity_item = entity_str.strip().split("<TCS>")
                
                if inerd_version == 1:
                    type = entity_item[0].strip()
                    entity = entity_item[1].strip()
                elif inerd_version == 2:
                    type = entity_item[1].strip()
                    entity = entity_item[0].strip()
                json_obj[type] = json_obj.get(type, []) + [entity]
            except Exception:
                raise json.JSONDecodeError("Invalid iNERD format.")
    return json_obj

def inerd_to_ordered_list(inerd_str, inerd_version=1):
    word_list = []
    type_list = []
    if len(inerd_str.strip()) > 0:
        entity_list = inerd_str.split("<ES>")
        for entity_str in entity_list:
            if len(entity_str.strip()) == 0:
                continue
            try:
                entity_item = entity_str.strip().split("<TCS>")
                
                if inerd_version == 1:
                    type = entity_item[0].strip()
                    entity = entity_item[1].strip()
                elif inerd_version == 2:
                    type = entity_item[1].strip()
                    entity = entity_item[0].strip()
                word_list.append(entity)
                type_list.append(type)
            except Exception:
                raise json.JSONDecodeError("Invalid iNERD format.")
    
    return word_list, type_list

def lists_to_ordered_inerd(ne_list, type_order, inerd_version=1):
    inerd_str = ""
    for idx in range(len(ne_list)):
        ner_type = type_order[idx]
        word = ne_list[idx]
        
        if inerd_version == 1:
            cur_entity_str = f"{ner_type} <TCS> {word} <ES> "
        elif inerd_version == 2:
            cur_entity_str = f"{word} <TCS> {ner_type} <ES> "
        
        inerd_str += cur_entity_str
    
    inerd_str = inerd_str.strip()
    return inerd_str

def dict_to_inerd(ner_dict):
    inerd_str = ""
    for ner_type, word_list in ner_dict.items():
        cur_entity_str = ""
        if inerd_str != "":
            inerd_str += " <ES> "
        for word in word_list:
            if cur_entity_str != "":
                cur_entity_str += " <ES> "
            
            cur_entity_str += f"{ner_type} <TCS> {word}"
        
        inerd_str += cur_entity_str
    
    if len(inerd_str) > 0:
        inerd_str += " <ES>"
    
    return inerd_str

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
def use_entity_list_for_entity_types_full(data_file_path, entity_types_dict_path, min_length_ratio=0.5, max_length_ratio=1.0, given_dataset_name=None, shuffle_list=True):
    import random
    import copy
    entity_lists = []
    df = pd.read_csv(data_file_path)
    
    data_format = None
    if 'inerd2' in data_file_path:
        data_format = 'inerd2'
    elif 'inerd' in data_file_path:
        data_format = 'inerd'
    else:
        data_format = 'json'
    
    with open(entity_types_dict_path, 'r', encoding='utf-8') as f:
        entity_types_dict = json.loads(f.read())

    for idx, row in df.iterrows():
        if given_dataset_name is not None:
            dataset_name = given_dataset_name
        else:
            dataset_name = row['types']
        entity_list = copy.deepcopy(entity_types_dict.get(dataset_name, []))
        
        if data_format == 'inerd':
            ner_str = row["NER"]
            if pd.isna(ner_str):
                ner_str = ""
            cur_json_dict = inerd_to_json(ner_str, inerd_version=1)
            word_list, type_list = inerd_to_ordered_list(ner_str, inerd_version=1)
        elif data_format == 'inerd2':
            ner_str = row["NER"]
            if pd.isna(ner_str):
                ner_str = ""
            cur_json_dict = inerd_to_json(ner_str, inerd_version=2)
            word_list, type_list = inerd_to_ordered_list(ner_str, inerd_version=2)
        elif data_format == 'json':
            cur_json_dict = json.loads(row["NER"])
        
        # 길이의 비율을 추출한다.
        length_ratio = random.uniform(min_length_ratio, max_length_ratio)
        sampled_entity_num = int(len(entity_list) * length_ratio)
        
        # 샘플링 자체에 shuffle 효과가 있음
        # 샘플해야 하는 수가 더 적을 때만 샘플링하기
        if sampled_entity_num < len(entity_list):
            # type 개수만큼 entity list에서 추출
            sampled_entities = random.sample(entity_list, sampled_entity_num)
            entity_list = sampled_entities

        if shuffle_list:
            # shuffle the entity list
            random.shuffle(entity_list)
        
        # Json을 인덱스 key 형태로 변경
        if data_format == 'json':
            json_dict_indexed = dict()
            for k in entity_list:
                if k in cur_json_dict:
                    json_dict_indexed[k] = cur_json_dict[k]
        else:
            cur_word_list = []
            cur_type_list = []
            for w, t in zip(word_list, type_list):
                if t in entity_list:
                    cur_word_list.append(w)
                    cur_type_list.append(t)
        
        entity_lists.append(" ".join(entity_list))
        if data_format == 'inerd':
            # inerd 형태로 변환
            inerd_str = lists_to_ordered_inerd(cur_word_list, cur_type_list, inerd_version=1)
            df.at[idx, "NER"] = inerd_str
        elif data_format == 'inerd2':
            inerd_str = lists_to_ordered_inerd(cur_word_list, cur_type_list, inerd_version=2)
            df.at[idx, "NER"] = inerd_str
        elif data_format == 'json':
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

def replace_test_labels(generated_file_path, test_file_path):
    gen_df = pd.read_csv(generated_file_path)
    test_df = pd.read_csv(test_file_path)

    if len(gen_df) != len(test_df):
        raise ValueError("Generated file and test file must have the same number of rows.")

    gen_df.loc[:, "NER"] = test_df["NER"]
    gen_df.to_csv(generated_file_path, index=False)

def make_curriculum_groups_for_learning(data_file_path, entity_list_file_path, dataset_name, inerd_version=1, shuffle_list=False):
    # 데이터셋에서 한 번에 여러 개의 entity type을 예측하는 대신에
    # 하나씩 생성해볼 수 있도록 그룹화한 데이터셋을 생성
    import random
    df = pd.read_csv(data_file_path)
    with open(entity_list_file_path, 'r', encoding='utf-8') as f:
        entity_types_dict = json.loads(f.read())
    
    curriculum_data_list = []
    df_len = len(df)
    
    progress_bar = tqdm(df.iterrows(), total=df_len, desc="Creating curriculum groups")
    for idx, row in progress_bar:
        # 한 가지 종류의 entity type을 순서대로 예측해보는 sample을 생성
        if not pd.isna(row["NER"]):
            json_dict = inerd_to_json(row["NER"], inerd_version=inerd_version)
            keys = list(json_dict.keys())
            
            for k in keys:
                word_list, type_list = inerd_to_ordered_list(row["NER"], inerd_version=inerd_version)
                
                cur_word_list = []
                cur_type_list = []
                for w, t in zip(word_list, type_list):
                    if t == k:
                        cur_word_list.append(w)
                        cur_type_list.append(t)
                    cur_entity_list = [k]
                
                    inerd_str = lists_to_ordered_inerd(cur_word_list, cur_type_list, inerd_version=inerd_version)
                curriculum_data_list.append({
                    "Sentence": row["Sentence"],
                    "NER": inerd_str,
                    "entity_list": " ".join(cur_entity_list),
                    "group_id": idx
                })
        
        # 그런 후에 현재 샘플의 데이터를 한꺼번에 생성해보는 sample을 생성
        # 전체를 한꺼번에 예측하는 sample은 별도의 그룹으로 묶기
        entity_types = entity_types_dict[dataset_name]
        if shuffle_list:
            random.shuffle(entity_types)
        curriculum_data_list.append({
            "Sentence": row["Sentence"],
            "NER": row["NER"],
            "entity_list": " ".join(entity_types),
            "group_id": df_len + 1
        })
        
    curriculum_df = pd.DataFrame(curriculum_data_list)
    return curriculum_df

def construct_contrastive_samples(data_file_path, entity_type, num_negatives=1, seed=42):
    import random
    import spacy
    
    random.seed(seed)
    df = pd.read_csv(data_file_path)
    contrastive_data_list = []
    nlp = spacy.load("en_core_web_sm")
    with open("/workspace/datas/entity_types.json", 'r', encoding='utf-8') as f:
        entity_types_dict = json.loads(f.read())
    
    entity_types = entity_types_dict.get(entity_type, [])
    assert len(entity_types) > 0, f"Entity type list for {entity_type} is empty."
    
    inerd_version = 1
    if "inerd2" in data_file_path:
        inerd_version = 2
    
    df_len = len(df)
    progress_bar = tqdm(df.iterrows(), total=df_len, desc="Creating contrastive samples")
    for idx, row in progress_bar:
        # 만약 NER 예측이 하나도 없는 문장이라면 본래 없는 NER span을 추가하는 negative sample 하나만 생성 (데이터셋 수를 짝수로 맞추기 위함)
        if pd.isna(row["NER"]):
            doc = nlp(row["Sentence"])
            # 명사 찾기
            noun_phrases = [chunk.text for chunk in doc.noun_chunks]
            # 만약 명사가 하나도 없다면 해당 sample을 완전히 건너뜀
            
            if len(noun_phrases) > 0:
                # Positive sample
                contrastive_data_list.append({
                    "Sentence": row["Sentence"],
                    "NER": row["NER"],
                    "contrastive_label": 1,
                    "group_id": idx
                })
                
                random_noun =  random.choice(noun_phrases) # 명사구 중 하나만 선택하여 ner로 할당
                word_list = [random_noun]
                type_list = [random.choice(entity_types)]
                
                inerd_str = lists_to_ordered_inerd(word_list, type_list, inerd_version=inerd_version)
                contrastive_data_list.append({
                    "Sentence": row["Sentence"],
                    "NER": inerd_str,
                    "contrastive_label": 0,
                    "group_id": idx
                })
        else:
            # NER이 있다면 평범하게 인수로 전달된 negative sample 수만큼 생성
            # Positive sample
            contrastive_data_list.append({
                "Sentence": row["Sentence"],
                "NER": row["NER"],
                "contrastive_label": 1,
                "group_id": idx
            })
            # Negative samples
            # 세 가지의 negative sample 종류
            # 1) span 중 하나의 entity type을 swap
            # 2) span을 밀거나 당기기거나 늘리기
            word_list, type_list = inerd_to_ordered_list(row["NER"], inerd_version=inerd_version)
            tag_list = inerd_to_tag_list(word_list, type_list, row["Sentence"])
            assert len(tag_list) == len(word_list), "Tag list and word list length mismatch."
            
            endurance = 50
            cur_trial = 0
            generated_negatives = 0
            generated_negative_samples_set = set()
            while generated_negatives < num_negatives and cur_trial < endurance:
                negative_word_list = []
                negative_type_list = []
                # 어떤 기법으로 negative sample을 생성할지 무작위로 선택
                negative_type = random.choice([1, 2])
                
                # 어떤 단어에 negative를 적용할지 선택
                target_idx = random.randint(0, len(word_list) - 1)
                
                if negative_type == 1:
                    # entity type swap
                    original_type = type_list[target_idx]
                    swapped_type = original_type
                    while swapped_type == original_type:
                        swapped_type = random.choice(entity_types)
                    
                    negative_word_list = word_list.copy()
                    negative_type_list = type_list.copy()
                    negative_type_list[target_idx] = swapped_type
                    
                    cur_neg_tag = tag_list[target_idx]
                    cur_neg_tag = (cur_neg_tag[0], cur_neg_tag[1], swapped_type)
                    if cur_neg_tag not in generated_negative_samples_set:
                        generated_negative_samples_set.add(cur_neg_tag)
                        
                        inerd_str = lists_to_ordered_inerd(negative_word_list, negative_type_list, inerd_version=inerd_version)
                        contrastive_data_list.append({
                            "Sentence": row["Sentence"],
                            "NER": inerd_str,
                            "contrastive_label": 0,
                            "group_id": idx
                        })
                        generated_negatives += 1
                        cur_trial = 0
                    else:
                        cur_trial += 1
                elif negative_type == 2:
                    # span을 한 칸씩 밀거나 당기기
                    pull_push = random.choice([(-1, -1), (1, 1), (0, 1), (-1, 0)])  # 앞으로 당기기, 뒤로 밀기, 뒤로 늘리기, 앞으로 늘리기
                   
                    cur_tag = tag_list[target_idx]
                    start_idx, end_idx, ner_type = cur_tag
                    
                    new_start_idx = start_idx + pull_push[0]
                    new_end_idx = end_idx + pull_push[1]
                    
                    if new_start_idx < 0 or new_end_idx > len(row["Sentence"].split()):
                        continue  # 문장 범위를 벗어나면 건너 뛰고 새로운 negative sample 시도
                    else:
                        negative_word_list = word_list.copy()
                        negative_type_list = type_list.copy()
                        
                        # 변경된 span 단어 추출
                        new_entity_words = row["Sentence"].split()[new_start_idx:new_end_idx]
                        new_entity_str = ' '.join(new_entity_words)
                        
                        negative_word_list[target_idx] = new_entity_str
                        
                        cur_neg_tag = (new_start_idx, new_end_idx, ner_type)
                        if cur_neg_tag not in generated_negative_samples_set:
                            generated_negative_samples_set.add(cur_neg_tag)
                            
                            inerd_str = lists_to_ordered_inerd(negative_word_list, negative_type_list, inerd_version=inerd_version)
                            contrastive_data_list.append({
                                "Sentence": row["Sentence"],
                                "NER": inerd_str,
                                "contrastive_label": 0,
                                "group_id": idx
                            })
                            generated_negatives += 1
                            cur_trial = 0
                        else:
                            cur_trial += 1
            
            if cur_trial >= endurance:
                # print(f"Warning: Could not generate enough negative samples for row {idx}. Generated {generated_negatives} out of {num_negatives}.")
                # 생성한 negative sample을 생성한 negative sample + positive sample이 짝수가 되도록 자름
                if generated_negatives % 2 == 0:
                    # positive sample이 한 개이므로 negative sample 수는 홀수가 되어야 함
                    contrastive_data_list = contrastive_data_list[:-1]
                
                
    
    contrastive_df = pd.DataFrame(contrastive_data_list)
    
    return contrastive_df