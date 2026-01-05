model_checkpoint_name=("ner-inerd2_conll2003-conll2003_contrastive_validonly_temp40_gen100_cont10_mixpool-fp32-w1e3-lr1e4-seed_42" "ner-inerd2_conll2003-conll2003_contrastive_validonly_temp40_gen100_cont20_mixpool-fp32-w1e3-lr1e4-seed_42" "ner-inerd2_conll2003-conll2003_contrastive_validonly_temp40_gen100_cont30_mixpool-fp32-w1e3-lr1e4-seed_42" "ner-inerd2_conll2003-conll2003_contrastive_validonly_temp40_gen100_cont40_mixpool-fp32-w1e3-lr1e4-seed_42" "ner-inerd2_conll2003-conll2003_contrastive_validonly_temp40_gen100_cont50_mixpool-fp32-w1e3-lr1e4-seed_42" "ner-inerd2_conll2003-conll2003_contrastive_validonly_temp40_gen100_cont60_mixpool-fp32-w1e3-lr1e4-seed_42" "ner-inerd2_conll2003-conll2003_contrastive_validonly_temp40_gen100_cont70_mixpool-fp32-w1e3-lr1e4-seed_42" "ner-inerd2_conll2003-conll2003_contrastive_validonly_temp40_gen100_cont80_mixpool-fp32-w1e3-lr1e4-seed_42" "ner-inerd2_conll2003-conll2003_contrastive_validonly_temp40_gen100_cont90_mixpool-fp32-w1e3-lr1e4-seed_42" "ner-inerd2_conll2003-conll2003_contrastive_validonly_temp40_gen100_cont100_mixpool-fp32-w1e3-lr1e4-seed_42")
model_name=("flan-t5-base" "flan-t5-base" "flan-t5-base" "flan-t5-base" "flan-t5-base" "flan-t5-base" "flan-t5-base" "flan-t5-base" "flan-t5-base" "flan-t5-base")
model_train_phrase=("second" "second" "second" "second" "second" "second" "second" "second" "second" "second")
dataset_names=("conll2003" "conll2003" "conll2003" "conll2003" "conll2003" "conll2003" "conll2003" "conll2003" "conll2003" "conll2003")
dataset_paths=("/workspace/datas/conll2003/testb.inerd2.csv" "/workspace/datas/conll2003/testb.inerd2.csv" "/workspace/datas/conll2003/testb.inerd2.csv" "/workspace/datas/conll2003/testb.inerd2.csv" "/workspace/datas/conll2003/testb.inerd2.csv" "/workspace/datas/conll2003/testb.inerd2.csv" "/workspace/datas/conll2003/testb.inerd2.csv" "/workspace/datas/conll2003/testb.inerd2.csv" "/workspace/datas/conll2003/testb.inerd2.csv" "/workspace/datas/conll2003/testb.inerd2.csv")
token_types=("inerd2" "inerd2" "inerd2" "inerd2" "inerd2" "inerd2" "inerd2" "inerd2" "inerd2" "inerd2")
index=0
while [ $index -lt 10 ]
do
    model_checkpoint_path="/workspace/model_dir/generation/${model_name[$index]}/${model_train_phrase[$index]}/parameter_search/${model_checkpoint_name[$index]}/final_model"

    python /workspace/ner_generation/generation_inference.py \
        --model_checkpoint ${model_checkpoint_path} \
        --output_file /workspace/datas/generated/${model_name[$index]}-${model_train_phrase[$index]}-${model_checkpoint_name[$index]}-tuned-${dataset_names[$index]}.csv \
        --batch_size 32 \
        --prompt_setting inerd2 \
        --dataset_name ${dataset_names[$index]} \
        --data_file ${dataset_paths[$index]} \
        --token_type ${token_types[$index]} \
        --gpus 2,3
    
    index=`expr $index + 1`
done
