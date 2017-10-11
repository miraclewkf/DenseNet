import os

# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ['MXNET_CPU_WORKER_NTHREADS'] = '2'
import mxnet as mx


# define metric of accuracy
class Accuracy(mx.metric.EvalMetric):
    def __init__(self, num=None):
        super(Accuracy, self).__init__('accuracy', num)

    def update(self, labels, preds):
        #mx.metric.check_label_shapes(labels, preds)

        #if self.num is not None:
        #    assert len(labels) == self.num

        pred_label = mx.nd.argmax_channel(preds[0]).asnumpy().astype('int32')
        label = labels[0].asnumpy().astype('int32')

        mx.metric.check_label_shapes(label, pred_label)

        self.sum_metric += (pred_label.flat == label.flat).sum()
        self.num_inst += len(pred_label.flat)





