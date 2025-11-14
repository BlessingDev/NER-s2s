
model_checkpoint="encoder-switch-ner-custom-class_weight-drop10-smoothing0-cycle10-lr2e-4-cosine_restart"
model_name="flan-t5-base"
dataset_method_name="switch"
dataset_name="wnut17"

python /workspace/encoder_binary/inference_classification_result.py \
    --model_checkpoint /workspace/model_dir/classification/${model_name}/${dataset_name}/${model_checkpoint}/final_model \
    --model_name ${model_name} \
    --prediction_output_dir /workspace/datas/encoder_result/${dataset_name}/${model_name}-${model_checkpoint} \
    --batch_size 256 \
    --custom_model \
    --decision_threshold 0.5 \
    --test_file /workspace/datas/wnut17/test.switch.csv