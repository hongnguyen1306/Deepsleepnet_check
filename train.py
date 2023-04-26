#! /usr/bin/python
# -*- coding: utf8 -*-
import os

import numpy as np
import argparse
import tensorflow as tf

from deepsleep.trainer import DeepFeatureNetTrainer, DeepSleepNetTrainer
from deepsleep.sleep_stage import (NUM_CLASSES,
                                   EPOCH_SEC_LEN,
                                   SAMPLING_RATE)


# Dùng kiểu này vì TensorFlow 2.x đã loại bỏ module tf.app.flags
# parser = argparse.ArgumentParser()
# parser.add_argument('--data_dir', type=str, default='data')
# parser.add_argument('--train_dir', type=str, default='train')
# parser.add_argument('--val_dir', type=str, default='val')
# parser.add_argument('--output_dir', type=str, default='output')
# parser.add_argument('--n_folds', type=int, default=20)
# parser.add_argument('--fold_idx', type=int, default=0)
# parser.add_argument('--pretrain_epochs', type=int, default=100)
# parser.add_argument('--finetune_epochs', type=int, default=200)
# parser.add_argument('--resume', type=bool, default=False)
# FLAGS, unparsed = parser.parse_known_args()

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', 'data',
                           """Directory where to load training data.""")
tf.app.flags.DEFINE_string('train_dir', 'data/physionet_sleep/train',
                           """Directory where to load training data.""")
tf.app.flags.DEFINE_string('val_dir', 'data/physionet_sleep/val',
                           """Directory where to load val data.""")
tf.app.flags.DEFINE_string('output_dir', '/output',
                           """Directory where to save trained models """
                           """and outputs.""")
tf.app.flags.DEFINE_integer('n_folds', 2,
                            """Number of cross-validation folds.""")
tf.app.flags.DEFINE_integer('fold_idx', 0,
                            """Index of cross-validation fold to train.""")
tf.app.flags.DEFINE_integer('pretrain_epochs', 100,
                            """Number of epochs for pretraining DeepFeatureNet.""")
tf.app.flags.DEFINE_integer('finetune_epochs', 200,
                            """Number of epochs for fine-tuning DeepSleepNet.""")
tf.app.flags.DEFINE_boolean('resume', False,
                            """Whether to resume the training process.""")


def pretrain(n_epochs):
    trainer = DeepFeatureNetTrainer(
        data_dir=FLAGS.data_dir,
        train_dir=FLAGS.train_dir,
        val_dir=FLAGS.val_dir,
        output_dir=FLAGS.output_dir,
        n_folds=2,
        fold_idx=FLAGS.fold_idx,
        batch_size=100,
        input_dims=EPOCH_SEC_LEN*100,
        n_classes=NUM_CLASSES,
        interval_plot_filter=50,
        interval_save_model=100,
        interval_print_cm=10
    )
    pretrained_model_path = trainer.train(
        n_epochs=n_epochs,
        resume=FLAGS.resume
    )
    return pretrained_model_path


def finetune(model_path, n_epochs):
    trainer = DeepSleepNetTrainer(
        data_dir=FLAGS.data_dir,
        train_dir=FLAGS.train_dir,
        val_dir=FLAGS.val_dir,
        output_dir=FLAGS.output_dir,
        n_folds=FLAGS.n_folds,
        fold_idx=FLAGS.fold_idx,
        batch_size=10,
        input_dims=EPOCH_SEC_LEN*100,
        n_classes=NUM_CLASSES,
        seq_length=25,
        n_rnn_layers=2,
        return_last=False,
        interval_plot_filter=50,
        interval_save_model=100,
        interval_print_cm=10
    )
    finetuned_model_path = trainer.finetune(
        pretrained_model_path=model_path,
        n_epochs=n_epochs,
        resume=FLAGS.resume
    )
    return finetuned_model_path


def main(argv=None):
    # Output dir
    output_dir = os.path.join(
        FLAGS.output_dir, "fold{}".format(FLAGS.fold_idx))
    if not FLAGS.resume:
        if tf.gfile.Exists(output_dir):
            tf.gfile.DeleteRecursively(output_dir)
        tf.gfile.MakeDirs(output_dir)

    pretrained_model_path = pretrain(
        n_epochs=FLAGS.pretrain_epochs
    )
    finetuned_model_path = finetune(
        model_path=pretrained_model_path,
        n_epochs=FLAGS.finetune_epochs
    )


if __name__ == "__main__":
    tf.compat.v1.app.run()
