
# 데이터 형태
# 단어, 품사, 구, NER

# 필요한 정보
# JSON 형태로 {원문: "", NER: {타입: [단어...]}}

import pandas as pd
import json

def preprocess_conll_data(file_path):
    """
    Preprocesses CoNLL data from a file and returns a DataFrame.
    
    Args:
        file_path (str): Path to the CoNLL formatted file.
        
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
    quote_count = 0
    for line in lines[1:]: # Skip the header line
        if line == '\n': # 문장 경계
            # 여기서는 문장 경계를 만나면 현재 문장과 NER 정보를 저장
            
            if len(cur_sentence) > 0:
                data_list.append({
                    "Sentence": " ".join(cur_sentence).strip(),
                    "NER": json.dumps(ner_list)
                })
                cur_sentence = []
                ner_list = dict()
                quote_count = 0
        else:
            # 각 줄에서 단어, 품사, 구, NER 정보를 추출
            word, pos, chunk, ner = line.strip().split()
            
            # 앞에 토큰에 붙이기
            if pos == "." or pos == "," or pos == ")" or pos == "POS":
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
            else:
                cur_sentence.append(word)
            
            # NER 정보 처리
            if ner != "O":
                ner = ner.split('-')[1]
                type_set.add(ner)
                if cur_ner is not None:
                    if cur_ner["type"] == ner:
                        cur_ner["words"].append(word)
                    else:
                        # 현재 NER 타입이 바뀌면 저장
                        if cur_ner["type"] not in ner_list:
                            ner_list[cur_ner["type"]] = [cur_ner["words"]]
                        else:
                            ner_list[cur_ner["type"]].append(cur_ner["words"])

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
            "원문": cur_sentence,
            "NER": json.dumps(ner_list)
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
            "원문": cur_sentence,
            "NER": json.dumps(ner_list)
        })

    df = pd.DataFrame(data_list)
    return df, list(type_set)

def preprocess_fewnerd_conll(file_path, focus_point="big"):
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
                
                data_list.append({
                    "Sentence": " ".join(cur_sentence).strip(),
                    "NER": json.dumps(ner_list)
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
        data_list.append({
            "Sentence": cur_sentence,
            "NER": json.dumps(ner_list)
        })

    df = pd.DataFrame(data_list)
    print("big entity type number: {0}".format(len(big_set)))
    print("small entity type number: {0}".format(len(small_set)))
    return df, list(big_set), list(small_set)