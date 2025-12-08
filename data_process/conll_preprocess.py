
# 데이터 형태
# 단어, 품사, 구, NER

# 필요한 정보
# JSON 형태로 {원문: "", NER: {타입: [단어...]}}

import pandas as pd
import json
from preprocess_utils import lists_to_ordered_inerd

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

def preprocess_conll_data(file_path, type_manner="simple"):
    """
    Preprocesses CoNLL data from a file and returns a DataFrame.
    
    Args:
        file_path (str): Path to the CoNLL formatted file.
        
    Returns:
        pd.DataFrame: DataFrame containing the preprocessed data.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    detail_type_map ={
        "LOC": "location",
        "PER": "person",
        "ORG": "organization",
        "MISC": "miscellaneous"
    }

    type_set = set()
    data_list = []
    cur_sentence = []
    ner_list = dict()
    cur_ner = None
    for line in lines[1:]: # Skip the header line
        if line == '\n': # 문장 경계
            # 여기서는 문장 경계를 만나면 현재 문장과 NER 정보를 저장
            # cur_ner에 정보가 들어있는지 확인
            if cur_ner is not None:
                if cur_ner["type"] not in ner_list:
                    ner_list[cur_ner["type"]] = [' '.join(cur_ner["words"])]
                else:
                    ner_list[cur_ner["type"]].append(' '.join(cur_ner["words"]))
            
            if len(cur_sentence) > 0:
                data_list.append({
                    "Sentence": " ".join(cur_sentence).strip(),
                    "NER": json.dumps(ner_list, ensure_ascii=False)
                })
            
            cur_ner = None
            cur_sentence = []
            ner_list = dict()
        else:
            # 각 줄에서 단어, 품사, 구, NER 정보를 추출
            word, pos, chunk, ner = line.strip().split()
            
            # 앞에 토큰에 붙이기
            '''if pos == "." or pos == "," or pos == ")" or pos == "POS":
                if len(cur_sentence) > 0:
                    cur_sentence[-1] += word
                else:
                    cur_sentence.append(word)
            elif len(cur_sentence) > 0 and (cur_sentence[-1] == "$" or cur_sentence[-1] == "(" or (cur_sentence[-1] == '"' and quote_count % 2 == 1)):
                cur_sentence[-1] += word
            elif pos == '"':
                quote_count += 1
                if quote_count % 2 == 0:
                    cur_sentence[-1] += word
                else:
                    cur_sentence.append(word)
            else:'''
            cur_sentence.append(word)
            
            # NER 정보 처리
            if ner != "O":
                ner = ner.split('-')[1]
                if type_manner == "detailed":
                    ner = detail_type_map.get(ner, ner)
                type_set.add(ner)
                if cur_ner is not None:
                    if cur_ner["type"] == ner:
                        cur_ner["words"].append(word)
                    else:
                        # 현재 NER 타입이 바뀌면 저장
                        if cur_ner["type"] not in ner_list:
                            ner_list[cur_ner["type"]] = [' '.join(cur_ner["words"])]
                        else:
                            ner_list[cur_ner["type"]].append(' '.join(cur_ner["words"]))

                        cur_ner = {"type": ner, "words": [word]}
                # 새로운 NER 타입 시작
                else:
                    cur_ner = {"type": ner, "words": [word]}
            else:
                # NER이 O인 경우 현재 NER 정보를 저장
                if cur_ner is not None:
                    if cur_ner["type"] not in ner_list:
                        ner_list[cur_ner["type"]] = [' '.join(cur_ner["words"])]
                    else:
                        ner_list[cur_ner["type"]].append(' '.join(cur_ner["words"]))
                    cur_ner = None
        
    # 마지막 문장 저장
    if cur_sentence:
        data_list.append({
            "Sentence": cur_sentence,
            "NER": json.dumps(ner_list, ensure_ascii=False)
        })

    df = pd.DataFrame(data_list)
    return df, list(type_set)

def preprocess_conll_data_to_ordered_inerd(file_path, type_manner="simple", inerd_version=1):
    """
    Preprocesses CoNLL data from a file and returns a DataFrame.
    
    Args:
        file_path (str): Path to the CoNLL formatted file.
        
    Returns:
        pd.DataFrame: DataFrame containing the preprocessed data.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    detail_type_map ={
        "LOC": "location",
        "PER": "person",
        "ORG": "organization",
        "MISC": "miscellaneous"
    }
    
    type_set = set()
    data_list = [] # DataFrame 구축을 위한 리스트
    cur_sentence = [] # 현재 문장 단어 모음
    ner_list = {
        "word_list": [], 
        "type_list": []
    } # 현재 문장에서 모으고 있는 NER 태그 정보
    cur_ner = None # 문장 안에서 현재 처리 중인 NER 정보
    for line in lines[1:]: # Skip the header line
        if line == '\n': # 문장 경계
            # 여기서는 문장 경계를 만나면 현재 문장과 NER 정보를 저장
            # cur_ner에 정보가 들어있는지 확인
            if cur_ner is not None:
                ner_list["word_list"].append(' '.join(cur_ner["words"]))
                ner_list["type_list"].append(cur_ner["type"])
            
            if len(cur_sentence) > 0:
                inerd_str = lists_to_ordered_inerd(ner_list["word_list"], ner_list["type_list"], inerd_version=inerd_version)
                data_list.append({
                    "Sentence": " ".join(cur_sentence).strip(),
                    "NER": inerd_str
                })
            
            cur_ner = None
            cur_sentence = []
            ner_list["word_list"] = []
            ner_list["type_list"] = []
        else:
            # 각 줄에서 단어, 품사, 구, NER 정보를 추출
            word, pos, chunk, ner = line.strip().split()
            
            # 앞에 토큰에 붙이기
            '''if pos == "." or pos == "," or pos == ")" or pos == "POS":
                if len(cur_sentence) > 0:
                    cur_sentence[-1] += word
                else:
                    cur_sentence.append(word)
            elif len(cur_sentence) > 0 and (cur_sentence[-1] == "$" or cur_sentence[-1] == "(" or (cur_sentence[-1] == '"' and quote_count % 2 == 1)):
                cur_sentence[-1] += word
            elif pos == '"':
                quote_count += 1
                if quote_count % 2 == 0:
                    cur_sentence[-1] += word
                else:
                    cur_sentence.append(word)
            else:'''
            cur_sentence.append(word)
            
            # NER 정보 처리
            if ner != "O":
                ner = ner.split('-')[1]
                if type_manner == "detailed":
                    ner = detail_type_map.get(ner, ner)
                type_set.add(ner)
                if cur_ner is not None:
                    if cur_ner["type"] == ner:
                        cur_ner["words"].append(word)
                    else:
                        # 현재 NER 타입이 바뀌면 저장
                        ner_list["word_list"].append(' '.join(cur_ner["words"]))
                        ner_list["type_list"].append(cur_ner["type"])

                        cur_ner = {"type": ner, "words": [word]}
                # 새로운 NER 타입 시작
                else:
                    cur_ner = {"type": ner, "words": [word]}
            else:
                # NER이 O인 경우 현재 NER 정보를 저장
                if cur_ner is not None:
                    # 현재 NER 타입이 바뀌면 저장
                    ner_list["word_list"].append(' '.join(cur_ner["words"]))
                    ner_list["type_list"].append(cur_ner["type"])
                    
                    cur_ner = None
        
    # loop를 빠져나왔음에도 문장이 존재한다면 저장
    if len(cur_sentence) > 0:
        inerd_str = lists_to_ordered_inerd(ner_list["word_list"], ner_list["type_list"], inerd_version=inerd_version)
        data_list.append({
            "Sentence": cur_sentence,
            "NER": inerd_str
        })

    df = pd.DataFrame(data_list)
    return df, list(type_set)

def preprocess_MIT_conll(file_path):
    """
    Preprocesses MIT CoNLL data from a file and returns a DataFrame.
    
    Args:
        file_path (str): Path to the MIT CoNLL formatted file.
        
    Returns:
        pd.DataFrame: DataFrame containing the preprocessed data.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    type_set = set()
    data_list = []
    cur_sentence = []
    ner_list = dict()
    cur_ner = None
    for line in lines:
        if line == '\n': # 문장 경계
            if len(cur_sentence) > 0:
                if cur_ner is not None:
                    if cur_ner["type"] not in ner_list:
                        ner_list[cur_ner["type"]] = [' '.join(cur_ner["words"])]
                    else:
                        ner_list[cur_ner["type"]].append(' '.join(cur_ner["words"]))
                    cur_ner = None
                
                data_list.append({
                    "Sentence": " ".join(cur_sentence).strip(),
                    "NER": json.dumps(ner_list)
                })
                cur_sentence = []
                ner_list = dict()
        else:
            # 각 줄에서 단어, 품사, 구, NER 정보를 추출
            ner, word = line.strip().split()
            
            cur_sentence.append(word)
            
            # NER 정보 처리
            if ner != "O":
                bi, type = ner.split('-')
                type_set.add(type)

                if cur_ner is not None:
                    if bi == 'I':
                        cur_ner["words"].append(word)
                    else:
                        # 현재 NER이 있는 상태에서 새로운 B를 만났을 때
                        if cur_ner["type"] not in ner_list:
                            ner_list[cur_ner["type"]] = [' '.join(cur_ner["words"])]
                        else:
                            ner_list[cur_ner["type"]].append(' '.join(cur_ner["words"]))

                        cur_ner = {"type": type, "words": [word]}
                else:
                    cur_ner = {"type": type, "words": [word]}
            else:
                if cur_ner is not None:
                    if cur_ner["type"] not in ner_list:
                        ner_list[cur_ner["type"]] = [' '.join(cur_ner["words"])]
                    else:
                        ner_list[cur_ner["type"]].append(' '.join(cur_ner["words"]))
                    cur_ner = None
        
    # 마지막 문장 저장
    if cur_sentence:
        data_list.append({
            "Sentence": cur_sentence,
            "NER": json.dumps(ner_list)
        })

    df = pd.DataFrame(data_list)
    return df, list(type_set)

def preprocess_MIT_conll_to_ordered_inerd(file_path, inerd_version=1):
    """
    Preprocesses MIT CoNLL data from a file and returns a DataFrame.
    
    Args:
        file_path (str): Path to the MIT CoNLL formatted file.
        
    Returns:
        pd.DataFrame: DataFrame containing the preprocessed data.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    type_set = set()
    data_list = []
    cur_sentence = []
    ner_list = {
        "word_list": [],
        "type_list": []
    }
    cur_ner = None
    for line in lines:
        if line == '\n': # 문장 경계
            if len(cur_sentence) > 0:
                if cur_ner is not None:
                    ner_list["word_list"].append(' '.join(cur_ner["words"]))
                    ner_list["type_list"].append(cur_ner["type"])
                    cur_ner = None
                
                ner_str = lists_to_ordered_inerd(ner_list["word_list"], ner_list["type_list"], inerd_version=inerd_version)
                
                data_list.append({
                    "Sentence": " ".join(cur_sentence).strip(),
                    "NER": ner_str
                })
                cur_sentence = []
                ner_list = {
                    "word_list": [],
                    "type_list": []
                }
        else:
            # 각 줄에서 단어, 품사, 구, NER 정보를 추출
            ner, word = line.strip().split()
            
            cur_sentence.append(word)
            
            # NER 정보 처리
            if ner != "O":
                bi, type = ner.split('-')
                type_set.add(type)

                if cur_ner is not None:
                    if bi == 'I':
                        cur_ner["words"].append(word)
                    else:
                        # 현재 NER이 있는 상태에서 새로운 B를 만났을 때
                        ner_list["word_list"].append(' '.join(cur_ner["words"]))
                        ner_list["type_list"].append(cur_ner["type"])

                        cur_ner = {"type": type, "words": [word]}
                else:
                    cur_ner = {"type": type, "words": [word]}
            else:
                if cur_ner is not None:
                    ner_list["word_list"].append(' '.join(cur_ner["words"]))
                    ner_list["type_list"].append(cur_ner["type"])
                    cur_ner = None
        
    # 마지막 문장 저장
    if cur_sentence:
        ner_str = lists_to_ordered_inerd(ner_list["word_list"], ner_list["type_list"], inerd_version=inerd_version)
        
        data_list.append({
            "Sentence": " ".join(cur_sentence).strip(),
            "NER": ner_str
        })

    df = pd.DataFrame(data_list)
    return df, list(type_set)

def preprocess_simple_conll(file_path, return_type="json"):
    """
    Preprocesses MIT CoNLL data from a file and returns a DataFrame.
    
    Args:
        file_path (str): Path to the MIT CoNLL formatted file.
        
    Returns:
        pd.DataFrame: DataFrame containing the preprocessed data.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    type_set = set()
    data_list = []
    cur_sentence = []
    ner_list = dict()
    cur_ner = None
    for line in lines:
        if line == '\n': # 문장 경계
            if len(cur_sentence) > 0:
                if cur_ner is not None:
                    if cur_ner["type"] not in ner_list:
                        ner_list[cur_ner["type"]] = [' '.join(cur_ner["words"])]
                    else:
                        ner_list[cur_ner["type"]].append(' '.join(cur_ner["words"]))
                    cur_ner = None
                
                ner_str = ""
                if return_type == "json":
                    ner_str = json.dumps(ner_list, ensure_ascii=False)
                elif return_type == "inerd":
                    ner_str = dict_to_inerd(ner_list)
                else:
                    raise ValueError("Invalid return_type. Choose 'json' or 'inerd'.")
                
                data_list.append({
                    "Sentence": " ".join(cur_sentence).strip(),
                    "NER": ner_str
                })
                cur_sentence = []
                ner_list = dict()
        else:
            # 각 줄에서 단어, 품사, 구, NER 정보를 추출
            word, ner = line.strip().split()
            
            cur_sentence.append(word)
            
            # NER 정보 처리
            if ner != "O":
                ner_divided = ner.split('-')
                bi = ner_divided[0]
                type = '-'.join(ner_divided[1:])
                type_set.add(type)

                if cur_ner is not None:
                    if bi == 'I':
                        cur_ner["words"].append(word)
                    else:
                        # 현재 NER이 있는 상태에서 새로운 B를 만났을 때
                        if cur_ner["type"] not in ner_list:
                            ner_list[cur_ner["type"]] = [' '.join(cur_ner["words"])]
                        else:
                            ner_list[cur_ner["type"]].append(' '.join(cur_ner["words"]))

                        cur_ner = {"type": type, "words": [word]}
                else:
                    cur_ner = {"type": type, "words": [word]}
            else:
                if cur_ner is not None:
                    if cur_ner["type"] not in ner_list:
                        ner_list[cur_ner["type"]] = [' '.join(cur_ner["words"])]
                    else:
                        ner_list[cur_ner["type"]].append(' '.join(cur_ner["words"]))
                    cur_ner = None
        
    # 마지막 문장 저장
    if cur_sentence:
        ner_str = ""
        if return_type == "json":
            ner_str = json.dumps(ner_list, ensure_ascii=False)
        elif return_type == "inerd":
            ner_str = dict_to_inerd(ner_list)
        
        data_list.append({
            "Sentence": " ".join(cur_sentence).strip(),
            "NER": ner_str
        })

    df = pd.DataFrame(data_list)
    return df, list(type_set)

def preprocess_simple_conll_to_ordered_inerd(file_path, inerd_version=1):
    """
    Preprocesses MIT CoNLL data from a file and returns a DataFrame.
    
    Args:
        file_path (str): Path to the MIT CoNLL formatted file.
        
    Returns:
        pd.DataFrame: DataFrame containing the preprocessed data.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    type_set = set()
    data_list = []
    cur_sentence = []
    ner_list = {
        "word_list": [], 
        "type_list": []
    } # 현재 문장에서 모으고 있는 NER 태그 정보
    cur_ner = None
    for line in lines:
        if line == '\n': # 문장 경계
            if len(cur_sentence) > 0:
                if cur_ner is not None:
                    ner_list["word_list"].append(' '.join(cur_ner["words"]))
                    ner_list["type_list"].append(cur_ner["type"])
                    cur_ner = None
                
                ner_str = lists_to_ordered_inerd(ner_list["word_list"], ner_list["type_list"], inerd_version=inerd_version)
                
                data_list.append({
                    "Sentence": " ".join(cur_sentence).strip(),
                    "NER": ner_str
                })
                cur_sentence = []
                ner_list = {
                    "word_list": [], 
                    "type_list": []
                }
        else:
            # 각 줄에서 단어, 품사, 구, NER 정보를 추출
            word, ner = line.strip().split()
            
            cur_sentence.append(word)
            
            # NER 정보 처리
            if ner != "O":
                ner_divided = ner.split('-')
                bi = ner_divided[0]
                type = '-'.join(ner_divided[1:])
                type_set.add(type)

                if cur_ner is not None:
                    if bi == 'I':
                        cur_ner["words"].append(word)
                    else:
                        # 현재 NER이 있는 상태에서 새로운 B를 만났을 때
                        ner_list["word_list"].append(' '.join(cur_ner["words"]))
                        ner_list["type_list"].append(cur_ner["type"])

                        cur_ner = {"type": type, "words": [word]}
                else:
                    cur_ner = {"type": type, "words": [word]}
            else:
                if cur_ner is not None:
                    ner_list["word_list"].append(' '.join(cur_ner["words"]))
                    ner_list["type_list"].append(cur_ner["type"])
                    cur_ner = None
        
    # 마지막 문장 저장
    if cur_sentence:
        ner_str = lists_to_ordered_inerd(ner_list["word_list"], ner_list["type_list"], inerd_version=inerd_version)
        
        data_list.append({
            "Sentence": " ".join(cur_sentence).strip(),
            "NER": ner_str
        })

    df = pd.DataFrame(data_list)
    return df, list(type_set)

def preprocess_fewnerd_to_ordered_inerd(file_path, focus_point="small", inerd_version=1):
    """
    Preprocesses FewNERD CoNLL data from a file and returns a DataFrame.
    
    Args:
        file_path (str): Path to the MIT CoNLL formatted file.
    
    Returns:
        pd.DataFrame: DataFrame containing the preprocessed data.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    big_set = set()
    small_set = set()

    data_list = []
    cur_sentence = []
    ner_list = {
        "word_list": [], 
        "type_list": []
    } # 현재 문장에서 모으고 있는 NER 태그 정보
    cur_ner = None
    for line in lines:
        if line == '\n': # 문장 경계
            if len(cur_sentence) > 0:
                if cur_ner is not None:
                    ner_list["word_list"].append(' '.join(cur_ner["words"]))
                    ner_list["type_list"].append(cur_ner["type"])
                    cur_ner = None
                
                ner_str = lists_to_ordered_inerd(ner_list["word_list"], ner_list["type_list"], inerd_version=inerd_version)
                
                data_list.append({
                    "Sentence": " ".join(cur_sentence).strip(),
                    "NER": ner_str
                })
                cur_sentence = []
                ner_list = {
                    "word_list": [], 
                    "type_list": []
                }
        else:
            # 각 줄에서 단어, 품사, 구, NER 정보를 추출
            word, ner = line.strip().split()
            
            cur_sentence.append(word)
            
            # NER 정보 처리
            if ner != "O":
                big, small = ner.split('-')
                big_set.add(big)
                small_set.add(ner)
                if focus_point == "big":
                    interested_ner_type = big
                else:
                    interested_ner_type = ner
                
                if cur_ner is not None:
                    if interested_ner_type == cur_ner["type"]:
                        cur_ner["words"].append(word)
                    else:
                        # 현재 NER이 있는 상태에서 새로운 B를 만났을 때
                        ner_list["word_list"].append(' '.join(cur_ner["words"]))
                        ner_list["type_list"].append(cur_ner["type"])

                        cur_ner = {"type": interested_ner_type, "words": [word]}
                else:
                    cur_ner = {"type": interested_ner_type, "words": [word]}
            else:
                if cur_ner is not None:
                    ner_list["word_list"].append(' '.join(cur_ner["words"]))
                    ner_list["type_list"].append(cur_ner["type"])
                    cur_ner = None
        
    # 마지막 문장 저장
    if cur_sentence:
        ner_str = lists_to_ordered_inerd(ner_list["word_list"], ner_list["type_list"], inerd_version=inerd_version)
        
        data_list.append({
            "Sentence": " ".join(cur_sentence).strip(),
            "NER": ner_str
        })

    df = pd.DataFrame(data_list)
    print("big entity type number: {0}".format(len(big_set)))
    print("small entity type number: {0}".format(len(small_set)))
    return df, list(big_set), list(small_set)

def preprocess_fewnerd_conll(file_path, focus_point="big", return_type="json"):
    """
    Preprocesses FewNERD CoNLL data from a file and returns a DataFrame.

    Args:
        file_path (str): Path to the MIT CoNLL formatted file.
        
    Returns:
        pd.DataFrame: DataFrame containing the preprocessed data.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    big_set = set()
    small_set = set()

    data_list = []
    cur_sentence = []
    ner_list = dict()
    cur_ner = None
    for line in lines:
        if line == '\n': # 문장 경계
            if len(cur_sentence) > 0:
                if cur_ner is not None:
                    if cur_ner["type"] not in ner_list:
                        ner_list[cur_ner["type"]] = [' '.join(cur_ner["words"])]
                    else:
                        ner_list[cur_ner["type"]].append(' '.join(cur_ner["words"]))
                    cur_ner = None
                
                ner_str = ""
                if return_type == "json":
                    ner_str = json.dumps(ner_list, ensure_ascii=False)
                elif return_type == "inerd":
                    ner_str = dict_to_inerd(ner_list)
                else:
                    raise ValueError("Invalid return_type. Choose 'json' or 'inerd'.")
                
                data_list.append({
                    "Sentence": " ".join(cur_sentence).strip(),
                    "NER": ner_str
                })
                cur_sentence = []
                ner_list = dict()
        else:
            # 각 줄에서 단어, 품사, 구, NER 정보를 추출
            word, ner = line.strip().split()
            
            cur_sentence.append(word)
            
            # NER 정보 처리
            if ner != "O":
                big, small = ner.split('-')
                big_set.add(big)
                small_set.add(ner)
                if focus_point == "big":
                    interested_ner_type = big
                else:
                    interested_ner_type = ner
                
                if cur_ner is not None:
                    if interested_ner_type == cur_ner["type"]:
                        cur_ner["words"].append(word)
                    else:
                        # 현재 NER이 있는 상태에서 새로운 B를 만났을 때
                        if cur_ner["type"] not in ner_list:
                            ner_list[cur_ner["type"]] = [' '.join(cur_ner["words"])]
                        else:
                            ner_list[cur_ner["type"]].append(' '.join(cur_ner["words"]))

                        cur_ner = {"type": interested_ner_type, "words": [word]}
                else:
                    cur_ner = {"type": interested_ner_type, "words": [word]}
            else:
                if cur_ner is not None:
                    if cur_ner["type"] not in ner_list:
                        ner_list[cur_ner["type"]] = [' '.join(cur_ner["words"])]
                    else:
                        ner_list[cur_ner["type"]].append(' '.join(cur_ner["words"]))
                    cur_ner = None
        
    # 마지막 문장 저장
    if cur_sentence:
        ner_str = ""
        if return_type == "json":
            ner_str = json.dumps(ner_list, ensure_ascii=False)
        elif return_type == "inerd":
            ner_str = dict_to_inerd(ner_list)
        else:
            raise ValueError("Invalid return_type. Choose 'json' or 'inerd'.")
        
        data_list.append({
            "Sentence": " ".join(cur_sentence).strip(),
            "NER": ner_str
        })

    df = pd.DataFrame(data_list)
    print("big entity type number: {0}".format(len(big_set)))
    print("small entity type number: {0}".format(len(small_set)))
    return df, list(big_set), list(small_set)

def json_dataset_to_inerd(file_path, json_column="NER"):
    df = pd.read_csv(file_path)
    for index, row in df.iterrows():
        ner_dict = json.loads(row[json_column])
        inerd_str = dict_to_inerd(ner_dict)
        df.at[index, json_column] = inerd_str
    
    return df


def preprocess_ace_conll_to_ordered_inerd(file_path, inerd_version=1, focus_point="small"):
    """
    Preprocesses ACE CoNLL data from a file and returns a DataFrame.
    
    Args:
        file_path (str): Path to the ACE CoNLL formatted file.
        return_type (str): Type of NER representation to return ("json" or "inerd").
    
    Returns:
        pd.DataFrame: DataFrame containing the preprocessed data.
    """
    ignore_types=["Sentence", "Crime", "Contact-Info", "Job-Title", "Numeric"]
    dataset_obj = None
    with open(file_path, 'r', encoding='utf-8') as file:
        dataset_obj = json.load(file)

    sentence_list = []
    ner_list = []
    ner_type_set = set()
    
    for row in dataset_obj:
        sentence = ' '.join(row["words"])
        entities = row["golden-entity-mentions"]
        
        entity_list = list()
        for entity in entities:
            ner_type = entity["entity-type"]
            if focus_point == "big":
                ner_type = ner_type.split(":")[0]
            
            if ner_type in ignore_types:
                continue
            
            ner_word = row["words"][entity["start"]:entity["end"]]
            ner_word = ' '.join(ner_word)
            ner_type_set.add(ner_type)
            
            start_idx = entity["start"]
            
            cur_tuple = (start_idx, ner_word, ner_type)
            if cur_tuple not in entity_list:
                entity_list.append(cur_tuple)
        
        # start index 기준으로 정렬
        entity_list_sorted = sorted(entity_list, key=lambda x: x[0])
        word_list = [item[1] for item in entity_list_sorted]
        type_list = [item[2] for item in entity_list_sorted]
        
        ner_str = lists_to_ordered_inerd(word_list, type_list, inerd_version=inerd_version)
        
        sentence_list.append(sentence)
        ner_list.append(ner_str)
    
    df = pd.DataFrame({
        "Sentence": sentence_list,
        "NER": ner_list
    })
    
    return df, list(ner_type_set)

def preprocess_genia_to_ordered_inerd(file_path, inerd_version=1):
    """
    Preprocesses Genia CoNLL data from a file and returns a DataFrame.
    
    Args:
        file_path (str): Path to the Genia CoNLL formatted file.
        
        inerd_version (int): Version of INERD format (1 or 2).
    Returns:
        pd.DataFrame: DataFrame containing the preprocessed data.
    """
    import ast
    
    dataset_df = pd.read_csv(file_path)
    
    type_set = set()
    data_list = []
    cur_sentence = []
    ner_list = {
        "word_list": [], 
        "type_list": []
    } # 현재 문장에서 모으고 있는 NER 태그 정보
    cur_ner = None
    for index, row in dataset_df.iterrows():
        tokens = ast.literal_eval(row["tokens"])
        entities = ast.literal_eval(row["entities"])
        
        # entity를 start index 기준으로 정렬
        entities_sorted = sorted(entities, key=lambda x: x['start'])
        
        # word_list와 type_list 구성
        for entity in entities_sorted:
            start = entity['start']
            end = entity['end']
            ner_type = entity['type']
            type_set.add(ner_type)
            
            word = ' '.join(tokens[start:end])
            ner_list["word_list"].append(word)
            ner_list["type_list"].append(ner_type)
        
        ner_str = lists_to_ordered_inerd(ner_list["word_list"], ner_list["type_list"], inerd_version=inerd_version)
        
        data_list.append({
            "Sentence": ' '.join(tokens).strip(),
            "NER": ner_str
        })
        
        ner_list = {
            "word_list": [], 
            "type_list": []
        } # NER 태그 정보 초기화
    
    df = pd.DataFrame(data_list)
    return df, list(type_set)

def preprocess_scierc_to_ordered_inerd(file_path, inerd_version=1):
    """
    Preprocesses SciERC CoNLL data from a file and returns a DataFrame.
    
    Args:
        file_path (str): Path to the SciERC CoNLL formatted file.
        
        inerd_version (int): Version of INERD format (1 or 2).
    Returns:
        pd.DataFrame: DataFrame containing the preprocessed data.
    """
    dataset_df = pd.read_json(file_path, lines=True)
    
    data_list = []
    type_set = set()
    for index, row in dataset_df.iterrows():
        sentences = row["sentences"]
        ners = row["ner"]
        
        offset = 0
        for sent_idx in range(len(sentences)):
            sentence = sentences[sent_idx]
            ner_info = ners[sent_idx]
            
            ner_list = {
                "word_list": [], 
                "type_list": []
            } # 현재 문장에서 모으고 있는 NER 태그 정보
            
            for ner in ner_info:
                start, end, ner_type = ner
                word = ' '.join(sentence[start - offset:end - offset + 1])
                
                type_set.add(ner_type)
                ner_list["word_list"].append(word)
                ner_list["type_list"].append(ner_type)
            
            ner_str = lists_to_ordered_inerd(ner_list["word_list"], ner_list["type_list"], inerd_version=inerd_version)
            
            data_list.append({
                "Sentence": ' '.join(sentence).strip(),
                "NER": ner_str
            })
            
            offset += len(sentence)
    
    df = pd.DataFrame(data_list)
    return df, list(type_set)