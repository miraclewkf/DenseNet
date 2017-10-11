python train.py --epoch 0 --model ./models/densenet-121 --batch-size 16 --num-classes 2 \
--data-train /your/path/to/train_data.lst --image-train /your/path/to/images/ \
--data-val /yout/path/to/test_data.lst --image-val /your/path/to/images/ \
--num-examples 20000 --lr 0.001 --gpus 0 --num-epoch 20 --save-result ./output/densenet-121 --save-name densenet-121