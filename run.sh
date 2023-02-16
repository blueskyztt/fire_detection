python3 prepro.py \
  --data_root /Users/${USERNAME}/Downloads/VOC2020 \
  --data_out_root ./datasets/fire
#  --demo

cd yolov5 || exit
KMP_DUPLICATE_LIB_OK=TRUE python3 train.py --data fire.yaml --epochs 100
weights_file="./runs/train/exp3/best.pt"
KMP_DUPLICATE_LIB_OK=TRUE python3 detect.py --weights ${weights_file} --source ../datasets/fire/images/val/0a96c6d2-ed93-4dec-8562-1c69af136ca7.jpg
