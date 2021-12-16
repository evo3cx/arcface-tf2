from absl import app, flags, logging
from absl.flags import FLAGS
import os
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from modules.models import ArcFaceModel
from modules.losses import SoftmaxLoss
from modules.utils import set_memory_growth, load_yaml, get_ckpt_inf
from modules import dataset


flags.DEFINE_string('cfg_path', '', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_string('train_dataset', '', 'path to binary dataset')
flags.DEFINE_integer('epoch', 5, 'num of epoch')


def main(_):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    print("flags", FLAGS.cfg_path)
    cfg = load_yaml(FLAGS.cfg_path)
    model = ArcFaceModel(size=cfg['input_size'],
                         backbone_type=cfg['backbone_type'],
                         num_classes=cfg['num_classes'],
                         head_type=cfg['head_type'],
                         embd_shape=cfg['embd_shape'],
                         w_decay=cfg['w_decay'],
                         training=True)
    model.summary(line_length=80)

    
    logging.info("load datasets.")
    dataset_len = cfg['num_samples']
    steps_per_epoch = dataset_len // cfg['batch_size']
    train_dataset = dataset.load_tfrecord_dataset(
        cfg['train_dataset'], cfg['batch_size'], cfg['binary_img'],
        is_ccrop=cfg['is_ccrop'])
    
    # set hyperparameter
    learning_rate = tf.constant(cfg['base_lr'])
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate, momentum=0.9, nesterov=True)
    loss_fn = SoftmaxLoss()
    
    # load checkpoints
    ckpt_path = tf.train.latest_checkpoint('./arcface-tf2/checkpoints/' + cfg['sub_name'])
    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        model.load_weights(ckpt_path)
        epochs, steps = get_ckpt_inf(ckpt_path, steps_per_epoch)
    else:
        print("[*] training from scratch.")
        epochs, steps = 1, 1

    
    model.compile(optimizer=optimizer, loss=loss_fn)

    mc_callback = ModelCheckpoint(
        './arcface-tf2/checkpoints/' + cfg['sub_name'] + '/e_{epoch}_b_{batch}.ckpt',
        save_freq=cfg['save_steps'] * cfg['batch_size'], verbose=1,
        save_weights_only=True)
    tb_callback = TensorBoard(log_dir='logs/',
                                update_freq=cfg['batch_size'] * 5,
                                profile_batch=0)
    tb_callback._total_batches_seen = steps
    tb_callback._samples_seen = steps * cfg['batch_size']
    callbacks = [mc_callback, tb_callback]

    history = model.fit(train_dataset,
                epochs=FLAGS.epoch,
                steps_per_epoch=steps_per_epoch,
                callbacks=callbacks,
                initial_epoch=epochs - 1)
    
    print(history.history)

    print("[*] training done!")


if __name__ == '__main__':
    app.run(main)
