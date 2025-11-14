dataset_name="wnut17"

python /workspace/encoder_binary/evaluate_with_boundary.py \
    --prediction_dir /workspace/datas/encoder_result/${dataset_name}/flan-t5-base-encoder-switch-ner-custom-class_weight-drop10-smoothing0-cycle10-lr2e-4-cosine_restart \
    --decision_threshold 0.9