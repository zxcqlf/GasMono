folder='path_to_weights'
#'tmp/kitti/para/test0/models'
path='path_to_nyu/testing/'

CUDA_VISIBLE_DEVICES=0 python evaluate_nyu.py --load_weights_folder $folder --eval_mono --data_path $path
