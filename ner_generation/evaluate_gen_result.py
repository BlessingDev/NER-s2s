import json
import argparse
import pandas as pd


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
        elif structure_type == "inerd":
            json_obj = inerd_to_json(json_str)
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
            

# 지표 2: JSON 파싱이 되었다면... 일반적인 NER 평가 지표(recall과 precision)을 사용
# NER 카테고리가 안 맞는 경우는 어떻게 할 것인가? Exact Match로 평가하자.
def evaluate_ner_f1(predictions, parsed_samples, args):
    # NER 평가 지표 계산
    # ...

    precision = 0.0
    recall = 0.0
    f1 = 0.0
    correct_num = 0
    gt_num = 0
    pd_num = 0

    if parsed_samples > 0:
        for index, row in predictions.iterrows():
            gt_json_str = row['NER']
            gt_parsed = parse_json(gt_json_str)

            json_str = row['generated_ner']
            pred_parsed = parse_json(json_str, args.structure_type)

            if pred_parsed is not None:
                watched_key = set()
                
                for k in gt_parsed.keys():
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
                        # 현재 키가 GT에는 없고, 예측에는 있다면
                        # 잘못된 것 개수 구하기
                        pd_num += len(pred_parsed[k])

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

    print("NER Evaluation Results:")
    for key, value in ner_json_results.items():
        print(f"  {key}: {value}")

    if ner_json_results["parsed_samples"] > 0:
        f1_results = evaluate_ner_f1(predictions, ner_json_results["parsed_samples"], args)
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
        choices=["json", "inerd", "simple_json"]
    )

    # args = parser.parse_args()
    args = parser.parse_args([
        "--prediction_file",
        "/workspace/datas/generated/gemma-3-270m-it-ner-inerd_ace05-fp32-w1e3-lr8e5-tuned-ace05.csv",
        "--structure_type", "inerd"
    ])
    
    main(args)