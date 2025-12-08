import conll_preprocess
import classification_preprocess
import preprocess_utils
import random
import json
import pathlib
import pandas as pd
import time

def main():
    print("Recon Bridging the Gap Module")
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    random.seed(42)
    
    split = "train"
    dataset_path = "conll2003"

    #df, big_ner_types, small_ner_types = conll_preprocess.preprocess_fewnerd_conll(f"/workspace/datas/fewnerd/supervised/{split}.txt", focus_point="small", return_type="json")
    #df, _, _ = conll_preprocess.preprocess_fewnerd_to_ordered_inerd(f"/workspace/datas/fewnerd/supervised/{split}.txt", focus_point="small", inerd_version=1)
    #df, ner_types = conll_preprocess.preprocess_simple_conll(f"/workspace/datas/{dataset_path}/{split}.txt", return_type="inerd")
    #df, ner_types = conll_preprocess.preprocess_simple_conll_to_ordered_inerd(f"/workspace/datas/{dataset_path}/{split}.txt", inerd_version=2)
    #df, ner_types = conll_preprocess.preprocess_conll_data(f"/workspace/datas/{dataset_path}/eng.{split}", type_manner="detailed")
    #df, ner_types = conll_preprocess.preprocess_conll_data_to_ordered_inerd(f"/workspace/datas/{dataset_path}/eng.{split}", type_manner="detailed", inerd_version=2)
    #df, ner_types = conll_preprocess.preprocess_MIT_conll_to_ordered_inerd(f"/workspace/datas/{dataset_path}/restaurant{split}.bio.txt", inerd_version=2)
    #df, ner_types = conll_preprocess.preprocess_ace_conll_to_ordered_inerd(f"/workspace/datas/{dataset_path}/{split}.json", inerd_version=2, focus_point="big")
    #df, ner_types = conll_preprocess.preprocess_genia_to_ordered_inerd(f"/workspace/datas/{dataset_path}/{split}.csv", inerd_version=2)
    #df, ner_types = conll_preprocess.preprocess_scierc_to_ordered_inerd(f"/workspace/datas/{dataset_path}/{split}.json", inerd_version=2)
    #df = classification_preprocess.preprocess_conll_to_binary(f"/workspace/datas/{dataset_path}/eng.{split}")
    #df = classification_preprocess.preprocess_conll_to_classification(f"/workspace/datas/{dataset_path}/eng.{split}")
    #df = classification_preprocess.mit_ner_to_classification(f"/workspace/datas/{dataset_path}/restaurant{split}.bio.txt")
    #df = classification_preprocess.preprocess_sim_conll_to_classification(f"/workspace/datas/{dataset_path}/{split}.txt", is_bio=True)
    #df = classification_preprocess.preprocess_classification_to_switch(f"/workspace/datas/{dataset_path}/{split}.classification.csv", "fewnerd_small", type_ratio=0.1, line_identifier="#/")
    
    #df = preprocess_utils.add_types_column_to_dataset(f"/workspace/datas/{dataset_path}/{split}.preprocessed.csv", "conll2003")
    #df = preprocess_utils.use_entity_list_for_entity_types_full(f"/workspace/datas/{dataset_path}/{split}.inerd2.csv", "/workspace/datas/entity_types.json", min_length_ratio=1.0, max_length_ratio=1.0, given_dataset_name="genia", shuffle_list=True)
    #df = conll_preprocess.json_dataset_to_inerd(f"/workspace/datas/{dataset_path}/{split}.json.random.csv", json_column="NER")
    '''df = preprocess_utils.make_curriculum_groups_for_learning(
        f"/workspace/datas/{dataset_path}/{split}.inerd2.csv",
        f"/workspace/datas/entity_types.json",
        dataset_name="genia",
        inerd_version=2
    )'''
    df = preprocess_utils.construct_contrastive_samples(f"/workspace/datas/{dataset_path}/{split}.inerd2.csv", entity_type="conll2003", num_negatives=7)
    
    df.to_csv(f"/workspace/datas/{dataset_path}/{split}.inerd2.contrastive2.csv", index=False)
    
    #preprocess_utils.split_train_dev(f"/workspace/datas/{dataset_path}/train.inerd2.csv", f"/workspace/datas/{dataset_path}/dev.inerd2.csv", dev_ratio=0.2)
    '''preprocess_utils.replace_test_labels(
        "/workspace/datas/generated/flan-t5-large-ner-inerd2_conll2003_detailed-ordered-conll2003-fp32-w1e3-lr1e4-tuned-conll2003_detailed.csv",
        "/workspace/datas/conll2003/testb.inerd2.detailed.csv"
    )'''

    #preprocess_utils.convert_ner_json_to_simplified_format(f"/workspace/datas/{dataset_path}/{split}.preprocessed.random.csv", f"/workspace/datas/{dataset_path}/{split}.preprocessed.random.sim.csv")

    # entity type 저장
    #preprocess_utils.save_entity_types(f"/workspace/datas/entity_types.json", "ace05_big", ner_types)

if __name__ == "__main__":
    main()