import argparse
import mxnet as mx
import os, sys
import logging

from metric import *

def get_fine_tune_model(sym, num_classes, layer_name):
    
    all_layers = sym.get_internals()
    net = all_layers[layer_name+'_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc')
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')

    return net

def multi_factor_scheduler(begin_epoch, epoch_size, step=[5,10], factor=0.1):
    step_ = [epoch_size * (x-begin_epoch) for x in step if x-begin_epoch > 0]
    return mx.lr_scheduler.MultiFactorScheduler(step=step_, factor=factor) if len(step_) else None

def train_model(model, gpus, batch_size, image_shape, epoch=0, num_epoch=20, kv='device'):
    train = mx.io.ImageRecordIter(
        path_imgrec         = args.data_train,
        label_width         = 1,
        mean_r              = 123.68,
        mean_g              = 116.779,
        mean_b              = 103.939,
        data_name           = 'data',
        label_name          = 'softmax_label',
        data_shape          = (3,224,224), 
        batch_size          = args.batch_size,
        rand_crop           = args.random_crop,
        rand_mirror         = args.random_mirror,
        shuffle             = True,
        num_parts           = kv.num_workers,
        resize              = 224,
        part_index          = kv.rank)

    val = mx.io.ImageRecordIter(
        path_imgrec         = args.data_val,
        label_width         = 1,
        mean_r              = 123.68,
        mean_g              = 116.779,
        mean_b              = 103.939,
        data_name           = 'data',
        label_name          = 'softmax_label',
        data_shape          = (3,224,224),
        batch_size          = args.batch_size,
        rand_crop           = False,
        rand_mirror         = False,      
        shuffle             = False,
        num_parts           = kv.num_workers,
        resize              = 224,
        part_index          = kv.rank)

    kv = mx.kvstore.create(args.kv_store)

    prefix = model
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

    new_sym = get_fine_tune_model(
        sym, args.num_classes, 'pool5')

    epoch_size = max(int(args.num_examples / args.batch_size / kv.num_workers), 1)
    lr_scheduler=multi_factor_scheduler(args.epoch, epoch_size)

    optimizer_params = {
            'learning_rate': args.lr,
            'momentum' : args.mom,
            'wd' : args.wd,
            'lr_scheduler': lr_scheduler}
    initializer = mx.init.Xavier(
            rnd_type='gaussian', factor_type="in", magnitude=2)

    if gpus == '':
        devs = mx.cpu()
    else:
        devs = [mx.gpu(int(i)) for i in gpus.split(',')]
        
    model = mx.mod.Module(
        context       = devs,
        symbol        = new_sym
    )

    checkpoint = mx.callback.do_checkpoint(args.save_result+args.save_name)

    eval_metric = mx.metric.CompositeEvalMetric()
    eval_metric.add(Accuracy())

    val_metric = mx.metric.CompositeEvalMetric()
    val_metric.add(Accuracy())

    model.fit(train,
              begin_epoch=epoch,
              num_epoch=num_epoch,
              eval_data=val,
              eval_metric=eval_metric,
              validation_metric=val_metric,
              kvstore=kv,
              optimizer='sgd',
              optimizer_params=optimizer_params,
              arg_params=arg_params,
              aux_params=aux_params,
              initializer=initializer,
              allow_missing=True, # for new fc layer
              batch_end_callback=mx.callback.Speedometer(args.batch_size, 20),
              epoch_end_callback=checkpoint)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='score a model on a dataset')
    parser.add_argument('--model',         type=str, required=True,)
    parser.add_argument('--gpus',          type=str, default='0')
    parser.add_argument('--batch-size',    type=int, default=200)
    parser.add_argument('--epoch',         type=int, default=0)
    parser.add_argument('--image-shape',   type=str, default='3,224,224')
    parser.add_argument('--data-train',    type=str)
    parser.add_argument('--data-val',      type=str)
    parser.add_argument('--num-classes',   type=int)
    parser.add_argument('--lr',            type=float, default=0.001)
    parser.add_argument('--num-epoch',     type=int, default=2)
    parser.add_argument('--kv-store',      type=str, default='device', help='the kvstore type')
    parser.add_argument('--save-result',   type=str, help='the save path')
    parser.add_argument('--num-examples',  type=int, default=20000)
    parser.add_argument('--mom',           type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--wd',            type=float, default=0.0001, help='weight decay for sgd')
    parser.add_argument('--save-name',     type=str, help='the save name of model')
    parser.add_argument('--random-crop',   type=int, default=1,help='if or not randomly crop the image')
    parser.add_argument('--random-mirror',   type=int, default=1,help='if or not randomly flip horizontally')
    args = parser.parse_args()

    kv = mx.kvstore.create(args.kv_store)

    if not os.path.exists(args.save_result):
        os.makedirs(args.save_result)

    # create a logger and set the level
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s')

    # this handler is used to record information in train.log
    hdlr = logging.FileHandler(args.save_result+ '/train.log')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    
    # this handler is used to print information in terminal
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    # record the information of args
    logging.info(args)

    train_model(model=args.model, gpus=args.gpus, batch_size=args.batch_size,
          image_shape='3,224,224', epoch=args.epoch, num_epoch=args.num_epoch, kv=kv)
