while getopts ':d:s:x:c:l:' opt
do
    case $opt in
        s)
        seed="$OPTARG" ;;
        x)
        max_seq_length="$OPTARG" ;;
        c)
        CUDA_IDS="$OPTARG" ;;
        l)
        lr="$OPTARG" ;;
        ?)
        exit 1;;
    esac
done



if [ ! "${max_seq_length}" ]
then 
  max_seq_length=-1
fi


gradient_clip_val=1
warmup_steps=100
weight_decay=0.01
precision=16

echo "lr=$lr; gradient_clip_val=${gradient_clip_val}; weight_decay=${weight_decay}; warmup_steps=${warmup_steps};"
echo "CUDA_VISIBLE_DEVICES=${CUDA_IDS}"


# --distributed_backend=ddp \

CUDA_VISIBLE_DEVICES=${CUDA_IDS} python train.py \
  --accelerator='ddp' \
  --gpus=1 \
  --precision=${precision} \
  --data_dir ../data/ \
  --output_dir ../output/ \
  --learning_rate ${lr}e-5 \
  --train_batch_size 32 \
  --eval_batch_size 32 \
  --seed ${seed} \
  --warmup_steps ${warmup_steps} \
  --lr_scheduler linear \
  --gradient_clip_val ${gradient_clip_val} \
  --weight_decay ${weight_decay} \
  --max_seq_length ${max_seq_length} \
  --max_epochs 10 \
  --val_check_interval 209 \
  --num_workers 32 \
  --do_train

  
  
  
