# CUDA_VISIBLE_DEVICES=1 python infer.py \
# \
#   --default_root_dir pred.csv \
#   --max_epochs 1 --batch_size 1 \
#   --lr 5e-5 --warmup_ratio 0.05 \
#   --num_workers 8 --devices 2 \
#   --dataset NICE --dropout 0.3 \
#   \
#   --enc_tok Salesforce/blip2-opt-2.7b \
#   --dec_tok Salesforce/blip2-opt-2.7b \
#   --model_card Salesforce/blip2-opt-2.7b
#     # --hparams \
#     # --model_path 

# 원래 path만 정해주면 infer하는데 
# 지금은 학습된 파라미터를 저장하지 않고 PLM 바로 쓰고 있기 때문에 이렇게
CUDA_VISIBLE_DEVICES=0 python infer.py \
\
  --default_root_dir pred2.csv \
  --max_epochs 1 --batch_size 1 \
  --lr 5e-5 --warmup_ratio 0.05 \
  --num_workers 8 --devices 2 \
  --dataset NICE --dropout 0.3 \
  \
  --enc_tok Salesforce/blip2-flan-t5-xl-coco \
  --dec_tok Salesforce/blip2-flan-t5-xl-coco \
  --model_card Salesforce/blip2-flan-t5-xl-coco
  #Salesforce/blip2-flan-t5-xl-coco