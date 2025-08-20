output_dir=out/stage1_pretraining

# deepspeed --include localhost:0,1,2,3 --master_port 29511 pre_training.py \
#    --batch-size 16 \
#    --gradient-accumulation-steps 8 \
#    --epochs 20 \
#    --opt AdamW \
#    --lr 3e-4 \
#    --quick_break 2048 \
#    --output_dir $output_dir \
#    --dataset CSL_News


# deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 29511 pre_training.py \
#    --batch-size 48 \
#    --gradient-accumulation-steps 2 \
#    --epochs 20 \
#    --opt AdamW \
#    --lr 3e-4 \
#    --quick_break 2048 \
#    --output_dir $output_dir \
#    --dataset CSL_News

nohup deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 29511 pre_training.py \
   --batch-size 48 \
   --gradient-accumulation-steps 2 \
   --epochs 20 \
   --opt AdamW \
   --lr 3e-4 \
   --quick_break 2048 \
   --output_dir out/stage1_pretraining \
   --dataset CSL_News \
   > training.log 2>&1 &
