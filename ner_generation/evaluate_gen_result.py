import json
import argparse
import pandas as pd
from tqdm.auto import tqdm


# 예측 csv 파일 불러오기
# 예측 csv에 정답도 포함됨
def load_data(prediction_file):
    predictions = pd.read_csv(prediction_file)

    return predictions

def inerd_to_json(inerd_str):
    json_obj = dict()
    if len(inerd_str.strip()) > 0:
        entity_list = inerd_str.split("<ES>")
        for entity_str in entity_list:
            if len(entity_str.strip()) == 0:
                continue
            try:
                entity_item = entity_str.strip().split("<TCS>")
                type = entity_item[0].strip()
                entity = entity_item[1].strip()
                json_obj[type] = json_obj.get(type, []) + [entity]
            except Exception:
                raise json.JSONDecodeError("Invalid iNERD format.")
    return json_obj
        
def inerd2_to_json(inerd_str):
    json_obj = dict()
    if len(inerd_str.strip()) > 0:
        entity_list = inerd_str.split("<ES>")
        for entity_str in entity_list:
            if len(entity_str.strip()) == 0:
                continue
            try:
                entity_item = entity_str.strip().split("<TCS>")
                type = entity_item[1].strip().replace("/ ", "/")
                entity = entity_item[0].strip()
                json_obj[type] = json_obj.get(type, []) + [entity]
                
                if len(entity) == 0 or len(type) == 0:
                    raise json.JSONDecodeError("Invalid iNERD format.")
            except Exception:
                raise json.JSONDecodeError("Invalid iNERD format.")
    return json_obj

# 지표 1: json이 제대로 파싱 되었는가.
# json 모듈로 파싱을 시켜보고, 파싱이 안 되면 해당 샘플에 0점
def parse_json(json_str, structure_type="json"):
    try:
        # json 문자열 전처리
        if pd.isna(json_str):
            json_str = ""
        if structure_type == "simple_json":
            json_str = '{' + json_str + '}'
            json_obj = json.loads(json_str)
        elif structure_type == "inerd1":
            json_obj = inerd_to_json(json_str)
        elif structure_type == "inerd2":
            json_obj = inerd2_to_json(json_str)
        else:
            json_obj = json.loads(json_str)
        
        return json_obj
    except (json.JSONDecodeError, TypeError):
        return None

def evaluate_ner_json(prediction_df, args):
    total_num = len(prediction_df)
    parsed_num = 0
    
    for index, row in prediction_df.iterrows():
        json_str = row['generated_ner']
        parsed = parse_json(json_str, args.structure_type)
        if parsed is not None:
            # JSON 파싱 성공
            parsed_num += 1
    
    return {
        "total_samples": total_num,
        "parsed_samples": parsed_num,
        "parsing_accuracy": parsed_num / total_num if total_num > 0 else 0
    }
            
def get_word_sequence_from_inerd_str(inerd_str, version=1):
    # inerd 파싱이 가능한 상태라고 가정 (예외 처리 없음)
    word_list = list()
    type_list = list()
    
    if pd.isna(inerd_str):
        return word_list, type_list
    
    entity_list = inerd_str.split("<ES>")
    for entity_str in entity_list:
        if len(entity_str.strip()) > 0:
            entity_item = entity_str.strip().split("<TCS>")
            if version == 1:
                entity = entity_item[1].strip()
                type = entity_item[0].strip()
            else:
                entity = entity_item[0].strip()
                type = entity_item[1].strip()
            
            #cur_word_list = entity.split()
            word_list.append(entity)
            type_list.append(type)
    
    return word_list, type_list

def inerd_to_tag_list(word_list, type_list, sentence, version=1):
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

# 지표 2: JSON 파싱이 되었다면... 일반적인 NER 평가 지표(recall과 precision)을 사용
# NER 카테고리가 안 맞는 경우는 어떻게 할 것인가? Exact Match로 평가하자.
def evaluate_ner_f1_exact_match(predictions, parsed_samples, args, entity_types=None):
    # NER 평가 지표 계산
    # ...

    precision = 0.0
    recall = 0.0
    f1 = 0.0
    correct_num = 0
    gt_num = 0
    pd_num = 0

    if parsed_samples > 0:
        progress_bar = tqdm(total=len(predictions), desc="Evaluating NER F1 (word match)")
        for index, row in predictions.iterrows():
            gt_json_str = row['NER']

            json_str = row['generated_ner']
            pred_parsed = parse_json(json_str, args.structure_type)

            gt_word_list, gt_type_list = get_word_sequence_from_inerd_str(gt_json_str, version=int(args.structure_type[-1]))
            
            gt_tags = inerd_to_tag_list(gt_word_list, gt_type_list, row['Sentence'], version=int(args.structure_type[-1]))
            
            gt_num += len(gt_tags)
            
            assert len(gt_tags) == len(gt_word_list), "Parsing error: GT tags length and word list length should be the same."
            
            if pred_parsed is not None:
                
                pd_word_list, pd_type_list = get_word_sequence_from_inerd_str(json_str, version=int(args.structure_type[-1]))
                
                pd_tags = inerd_to_tag_list(pd_word_list, pd_type_list, row['Sentence'], version=int(args.structure_type[-1]))
                
                #assert len(pd_tags) == len(pd_word_list), "Parsing error: prediction tags length and word list length should be the same."
                
                for pd_tag in pd_tags:
                    pd_num += 1
                    if pd_tag in gt_tags:
                        correct_num += 1
                
                '''
                gt_parsed = parse_json(gt_json_str, args.structure_type)watched_key = set()
                
                for k in gt_parsed.keys():
                    if entity_types is not None and k not in entity_types:
                        continue
                    watched_key.add(k)
                    gt_num += len(gt_parsed[k])
                    if k in pred_parsed:
                        # 만약 현재 키가 예측 json에도 들어있다면
                        # 맞춘 것 개수 구하기
                        pd_num += len(pred_parsed[k])
                        for v in gt_parsed[k]:
                            if v in pred_parsed[k]:
                                correct_num += 1
                
                for k in pred_parsed.keys():
                    if k not in watched_key:
                        if entity_types is not None and k not in entity_types:
                            continue
                        # 현재 키가 GT에는 없고, 예측에는 있다면
                        # 잘못된 것 개수 구하기
                        pd_num += len(pred_parsed[k])'''
                
            progress_bar.update(1)
        progress_bar.close()

    precision = correct_num / pd_num if pd_num > 0 else 0
    recall = correct_num / gt_num if gt_num > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("Correct:", correct_num, "Predicted:", pd_num, "Ground Truth:", gt_num)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def evaluate_ner_f1_word_match(predictions, parsed_samples, args, tokenizer=None):
    # NER 평가 지표 계산
    # 개수 매칭이 아니라 토큰 매칭으로 평가하기

    precision = 0.0
    recall = 0.0
    f1 = 0.0
    correct_num = 0
    gt_num = 0
    pd_num = 0

    progress_bar = tqdm(total=len(predictions), desc="Evaluating NER F1 (word match)")
    if parsed_samples > 0:
        for index, row in predictions.iterrows():
            gt_json_str = row['NER']

            json_str = row['generated_ner']
            pred_parsed = parse_json(json_str, args.structure_type)

            if pred_parsed is not None and pd.isna(gt_json_str) == False:
                gt_words, gt_types = get_word_sequence_from_inerd_str(gt_json_str, version=int(args.structure_type[-1]))
                pd_words, pd_types = get_word_sequence_from_inerd_str(json_str, version=int(args.structure_type[-1]))
                
                pd_index = 0
                for w, t in zip(gt_words, gt_types):
                    if tokenizer is not None:
                        gt_tokens = tokenizer(w, add_special_tokens=False)
                        cur_token_len = len(gt_tokens['input_ids'])
                        gt_num += cur_token_len # 토큰 단위로 개수 셈
                    else:
                        gt_num += 1 # 단어 단위로 개수 셈
                    
                    if w in pd_words[pd_index:]:
                        while pd_index < len(pd_words) and pd_words[pd_index] != w:
                            pd_index += 1
                        if pd_index < len(pd_words) and pd_words[pd_index] == w:
                            # 단어 매칭 성공
                            if t == pd_types[pd_index]:
                                if tokenizer is not None:
                                    correct_num += cur_token_len
                                else:
                                    correct_num += 1
                            pd_index += 1
                    
                
                if tokenizer is not None:
                    for w in pd_words:
                        pd_tokens = tokenizer(w, add_special_tokens=False)
                        pd_num += len(pd_tokens['input_ids'])
                else:
                    pd_num += len(pd_words)
            
            progress_bar.update(1)
    
    progress_bar.close()
                
                    

    precision = correct_num / pd_num if pd_num > 0 else 0
    recall = correct_num / gt_num if gt_num > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# Main 함수
def main(args):
    predictions = load_data(args.prediction_file)

    ner_json_results = evaluate_ner_json(predictions, args)

    f1_results = dict()
    if ner_json_results["parsed_samples"] > 0:
        if args.tokenizer_path is not None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        else:
            tokenizer = None
        
        f1_results = evaluate_ner_f1_exact_match(predictions, ner_json_results["parsed_samples"], args, entity_types=None)
        
        
    print("NER Evaluation Results:")
    for key, value in ner_json_results.items():
        print(f"  {key}: {value}")
    print("NER Evaluation Results (without category):")
    for key, value in f1_results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NER JSON generation results.")
    
    parser.add_argument(
        "--prediction_file",
        required=True,
        type=str
    )
    parser.add_argument(
        "--structure_type",
        type=str,
        default="json",
        choices=["json", "inerd1", "inerd2", "simple_json"]
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None
    )

    # args = parser.parse_args()
    args = parser.parse_args([
        "--prediction_file",
        "/workspace/datas/generated/flan-t5-base-second-ner-inerd2_conll2003-conll2003_contrastive_validonly_temp40_gen100_cont100_mixpool-fp32-w1e3-lr1e4-seed_42-tuned-conll2003.csv",
        "--structure_type", "inerd2",
#        "--tokenizer_path" , "/workspace/model_dir/reference/ontonotes-plmarker"
    ])
    
    print(args)
    main(args)