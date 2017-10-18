python train_rec.py --batch-size 16 --num-classes 2 --epoch 0 --model model/densenet-169 \
--data-train /your/path/to/train_data.rec \
--data-val /your/path/to/val_data.rec \
--num-examples 20000 --lr 0.001 --gpus 0 --num-epoch 20 --save-result output/test/ --save-name test


