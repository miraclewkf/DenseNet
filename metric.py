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

class Class3_acc(mx.metric.EvalMetric):
    def __init__(self):
        super(Class3_acc, self).__init__('Class3_acc')
    def update(self, labels, preds):
        pred = mx.nd.argmax_channel(preds[0]).asnumpy().astype('int32')
        label = labels[0].asnumpy().astype('int32')

        mx.metric.check_label_shapes(label, pred)

        count = 0
        for num1 in range(len(pred.flat)):
            if pred[num1] >= 0 and pred[num1] < 3 and label[num1] >= 0 and label[num1] < 3: # nudity
                count +=1
            elif pred[num1] == 3 and label[num1] == 3:
                count +=1
            elif (pred[num1] == 4 or pred[num1] == 5) and (label[num1] == 4 or label[num1] ==5):
                count +=1

        self.sum_metric += count
        self.num_inst += len(pred.flat)

class Recall_nudity(mx.metric.EvalMetric):
    def __init__(self):
        super(Recall_nudity, self).__init__('Recall_nudity')
    def update(self, labels, preds):
        pred = mx.nd.argmax_channel(preds[0]).asnumpy().astype('int32')
        label = labels[0].asnumpy().astype('int32')

        mx.metric.check_label_shapes(label, pred)

        count_nudity = 0
        truth_nudity = 0
        for num1 in range(len(pred.flat)):
            if pred[num1] >= 0 and pred[num1] < 3 and label[num1] >= 0 and label[num1] < 3: # nudity
                count_nudity +=1
            if label[num1] >=0 and label[num1] < 3:
                truth_nudity +=1

        self.sum_metric += count_nudity
        self.num_inst += truth_nudity

class Recall_nudity_level1(mx.metric.EvalMetric):
    def __init__(self):
        super(Recall_nudity_level1, self).__init__('Recall_nudity_level1')
    def update(self, labels, preds):
        pred = mx.nd.argmax_channel(preds[0]).asnumpy().astype('int32')
        label = labels[0].asnumpy().astype('int32')

        mx.metric.check_label_shapes(label, pred)

        count_nudity = 0
        truth_nudity = 0
        for num1 in range(len(pred.flat)):
            if pred[num1] == 0 and label[num1] == 0: # nudity level1
                count_nudity +=1
            if label[num1] ==0:
                truth_nudity +=1

        self.sum_metric += count_nudity
        self.num_inst += truth_nudity

class Precision_nudity(mx.metric.EvalMetric):
    def __init__(self):
        super(Precision_nudity, self).__init__('Precision_nudity')
    def update(self, labels, preds):
        pred = mx.nd.argmax_channel(preds[0]).asnumpy().astype('int32')
        label = labels[0].asnumpy().astype('int32')

        mx.metric.check_label_shapes(label, pred)

        count_nudity = 0
        pred_nudity = 0
        for num1 in range(len(pred.flat)):
            if pred[num1] >= 0 and pred[num1] < 3 and label[num1] >= 0 and label[num1] < 3: # nudity
                count_nudity +=1
            if pred[num1] >=0 and pred[num1] < 3:
                pred_nudity +=1

        self.sum_metric += count_nudity
        self.num_inst += pred_nudity

class Precision_nudity_level1(mx.metric.EvalMetric):
    def __init__(self):
        super(Precision_nudity_level1, self).__init__('Precision_nudity_level1')
    def update(self, labels, preds):
        pred = mx.nd.argmax_channel(preds[0]).asnumpy().astype('int32')
        label = labels[0].asnumpy().astype('int32')

        mx.metric.check_label_shapes(label, pred)

        count_nudity = 0
        pred_nudity = 0
        for num1 in range(len(pred.flat)):
            if pred[num1] == 0 and label[num1] == 0: # nudity level1
                count_nudity +=1
            if pred[num1] ==0:
                pred_nudity +=1

        self.sum_metric += count_nudity
        self.num_inst += pred_nudity

class Fscore_nudity(mx.metric.EvalMetric):
    def __init__(self):
        super(Fscore_nudity, self).__init__('Fscore_nudity')
    def update(self, labels, preds):
        pred = mx.nd.argmax_channel(preds[0]).asnumpy().astype('int32')
        label = labels[0].asnumpy().astype('int32')

        mx.metric.check_label_shapes(label, pred)

        count_nudity = 0
        truth_nudity = 0
        predict_nudity = 0
        for num1 in range(len(pred.flat)):
            if pred[num1] >= 0 and pred[num1] < 3 and label[num1] >= 0 and label[num1] < 3: # nudity
                count_nudity +=1
            if label[num1] >=0 and label[num1] < 3:
                truth_nudity +=1
            if pred[num1] >= 0 and pred[num1] < 3:
                predict_nudity +=1
        Recall_nudity = float(count_nudity)/float(truth_nudity+0.000001)
        Precision_nudity = float(count_nudity)/float(predict_nudity+0.000001)

        self.sum_metric += (2*Recall_nudity*Precision_nudity)
        self.num_inst += (Recall_nudity+Precision_nudity)

class Fscore_nudity_level1(mx.metric.EvalMetric):
    def __init__(self):
        super(Fscore_nudity_level1, self).__init__('Fscore_nudity_level1')
    def update(self, labels, preds):
        pred = mx.nd.argmax_channel(preds[0]).asnumpy().astype('int32')
        label = labels[0].asnumpy().astype('int32')

        mx.metric.check_label_shapes(label, pred)

        count_nudity = 0
        truth_nudity = 0
        predict_nudity = 0
        for num1 in range(len(pred.flat)):
            if pred[num1] == 0 and label[num1] == 0: # nudity level1
                count_nudity +=1
            if label[num1] ==0:
                truth_nudity +=1
            if pred[num1] == 0:
                predict_nudity +=1
        Recall_nudity = float(count_nudity)/float(truth_nudity+0.000001)
        Precision_nudity = float(count_nudity)/float(predict_nudity+0.000001)

        self.sum_metric += (2*Recall_nudity*Precision_nudity)
        self.num_inst += (Recall_nudity+Precision_nudity)




