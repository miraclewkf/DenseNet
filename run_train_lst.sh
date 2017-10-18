python train_lst.py --batch-size 16 --num-classes 2 --epoch 0 --model model/densenet-169 \
--data-train /your/path/to/train_data.lst --image-train /your/path/to/images/ \
--data-val /yout/path/to/val_data.lst --image-val /your/path/to/images/ \
--num-examples 20000 --lr 0.001 --gpus 0 --num-epoch 20 --save-result output/test/ --save-name test

