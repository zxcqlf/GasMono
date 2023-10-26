# run the code on indoor set
#NYUV2
path='path/to/nyu/training'
gpus=0
#sleep 1m
CUDA_VISIBLE_DEVICES=$gpus python train.py --model_name tmp_t0 --split nyu --dataset nyu --height 256 --width 320 --data_path $path --learning_rate 5e-5 --use_posegt --www 0.2 --wpp 0.2 --iiters 2 --selfpp --batch_size 12 --disparity_smoothness 1e-4

