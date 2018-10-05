#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: svhn-digit-dorefa.py
# Author: Yuxin Wu

import argparse
import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.dataflow import dataset
from tensorpack.tfutils.varreplace import remap_variables
import os

from dorefa import get_dorefa

"""
This is a tensorpack script for the SVHN results in paper:
DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients
http://arxiv.org/abs/1606.06160

The original experiements are performed on a proprietary framework.
This is our attempt to reproduce it on tensorpack.

Accuracy:
    With (W,A,G)=(1,1,4), can reach 3.1~3.2% error after 150 epochs.
    With the GaussianDeform augmentor, it will reach 2.8~2.9%
    (we are not using this augmentor in the paper).

    With (W,A,G)=(1,2,4), error is 3.0~3.1%.
    With (W,A,G)=(32,32,32), error is about 2.9%.

Speed:
    30~35 iteration/s on 1 TitanX Pascal. (4721 iterations / epoch)

To Run:
    ./svhn-digit-dorefa.py --dorefa 1,2,4
"""

BITW = 1
BITA = 2
BITG = 4


class Model(ModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.float32, [None, 32, 32, 3], 'input'),
                tf.placeholder(tf.int32, [None], 'label'),
                tf.placeholder(tf.float32, (), 'relax')]

    def build_graph(self, image, label):
        is_training = get_current_tower_context().is_training
        print('WAG: ', BITW, BITA, BITG)
        fw, fa, fg = get_dorefa(BITW, BITA, BITG)

        # monkey-patch tf.get_variable to apply fw
        def binarize_weight(v):
            name = v.op.name
            # don't binarize first and last layer
            if not name.endswith('W') or 'conv0' in name or 'fc2' in name:
                return v
            else:
                logger.info("Binarizing weight {}".format(v.op.name))
                return fw(v)

        def activate(x):
            return fa(x)

        keep_prob = tf.constant(0.5 if is_training else 1.0)

        if is_training:
            tf.summary.image("train_image", image, 10)
        if tf.test.is_gpu_available():
            image = tf.transpose(image, [0, 3, 1, 2])
            data_format = 'channels_first'
        else:
            data_format = 'channels_last'

        image = image / 4.0     # just to make range smaller

        with remap_variables(binarize_weight), \
                argscope(BatchNorm, momentum=0.9, epsilon=1e-4, center=False, scale=False), \
                argscope(Conv2D, use_bias=False, kernel_size=3),\
                argscope([Conv2D, MaxPooling, BatchNorm], data_format=data_format):
            logits = (LinearWrap(image)
                      .Conv2D('conv1.1', filters=64)
                      .BatchNorm('bn0')
                      .apply(activate)
                      # 18
                      .Conv2D('conv1.2', filters=64)
                      .apply(fg)
                      .BatchNorm('bn1').apply(activate)
                      .MaxPooling('pool1', 3, stride=2, padding='SAME') 

                      .Conv2D('conv2.1', filters=128)
                      .apply(fg)
                      .BatchNorm('bn2').apply(activate)
                      # 9
                      .Conv2D('conv2.2', filters=128)
                      .apply(fg)
                      .BatchNorm('bn3').apply(activate)
                      .MaxPooling('pool1', 3, stride=2, padding='SAME') 
                      # 7

                      .Conv2D('conv3.1' , filters=128, padding='VALID')
                      .apply(fg)
                      .BatchNorm('bn4').apply(activate)

                      .Conv2D('conv3.2' , filters=128, padding='VALID')
                      .apply(fg)
                      .BatchNorm('bn5').apply(activate)
                      # 5
                      .FullyConnected('fc0', 1024 + 512)
                      .apply(fg)
                      .BatchNorm('bn6').apply(activate)
                      .tf.nn.dropout(keep_prob)
                      .FullyConnected('fc1', 512) 
                      .apply(fg)
                      .BatchNorm('bn7').apply(cabs)
                      .FullyConnected('fc2', 10)())
        #tf.nn.softmax(logits, name='output')

        # compute the number of failed samples
        wrong = tf.cast(tf.logical_not(tf.nn.in_top_k(logits, label, 1)), tf.float32, name='wrong_tensor')
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')
        # weight decay on all W of fc layers
        wd_cost = regularize_cost('fc.*/W', l2_regularizer(1e-7))

        add_param_summary(('.*/W', ['histogram', 'rms']))
        total_cost = tf.add_n([cost, wd_cost], name='cost')
        add_moving_summary(cost, wd_cost, total_cost)
        return total_cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.01, trainable=False)
        # opt = tf.train.MomentumOptimizer(lr, 0.9)
        opt = tf.train.AdamOptimizer(lr)
        tf.summary.scalar('lr', lr)
        return opt


def get_data(train_or_test, dir):
    BATCH_SIZE = 128
    isTrain = train_or_test == 'train'
    ds = dataset.Cifar10(train_or_test, dir=dir)
    pp_mean = ds.get_per_pixel_mean()
    if isTrain:
        augmentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
            imgaug.MapImage(lambda x: x - pp_mean),
        ]
    else:
        augmentors = [
            imgaug.MapImage(lambda x: x - pp_mean)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds+


def get_config():
    #logger.auto_set_dir()
    logger.set_logger_dir('../../../runs/'+args.root_dir)
    # prepare dataset
    dataset_train = get_data('train', dir = '../../../cifar10_data/')
    dataset_test = get_data('test', dir = '../../../cifar10_data/')

    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test,
                            [ScalarStats('cost'), ClassificationError('wrong_tensor')]),
            ScheduledHyperParamSetter('learning_rate',
                                      [(1, 0.01), (82, 0.001), (123, 0.0002), (200, 0.0001)])
        ],
        model=Model(),
        max_epoch=300,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dorefa',
                        help='number of bits for W,A,G, separated by comma. Defaults to \'1,2,4\'',
                        default='32,2,32')
    parser.add_argument('--root_dir', action='store', default='trash/', help='root dir for different experiments',
    					type=str)
    parser.add_argument('--gpu', help='the physical ids of GPUs to use')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    BITW, BITA, BITG = map(int, args.dorefa.split(','))
    config = get_config()
    launch_train_with_config(config, SimpleTrainer())
