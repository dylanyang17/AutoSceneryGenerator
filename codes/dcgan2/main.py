import tensorflow as tf
import os
from model import DCGAN

flags = tf.app.flags
flags.DEFINE_integer("epoch", 800, "Epoch")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate")
flags.DEFINE_integer("batch_size", 64, "Batch size")
flags.DEFINE_string("dataset", "mountains", "The name of dataset")
flags.DEFINE_string("data_dir", "./data", "path to datasets")
flags.DEFINE_integer("z_dim", 100, "dimensions of z")
flags.DEFINE_integer("max_to_keep", 100, "maximum number of checkpoints to keep")
flags.DEFINE_integer("sample_freq", 200, "sample every this many iterations")
flags.DEFINE_integer("ckpt_freq", 200, "save checkpoint every this many iterations")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Checkpoint dir")
flags.DEFINE_string("sample_dir", "samples", "Sample dir")
flags.DEFINE_integer("input_height", 256, "Input height")
flags.DEFINE_integer("input_width", None, "If None, same value as input_height")
flags.DEFINE_integer("output_height", 64, "Output height")
flags.DEFINE_integer("output_width", None, "If None, same value as output_height")
flags.DEFINE_string("out_dir", "./out", "Directory for outputs ")
flags.DEFINE_float("beta1", 0.5, "Momentum")
flags.DEFINE_string("out_name", "", "")

FLAGS = flags.FLAGS

# os.environ["CUDA_VISIBLE_DEVICES"]="0"


def main(_):
    
    if FLAGS.output_height is None: FLAGS.output_height = FLAGS.input_height
    if FLAGS.input_width is None: FLAGS.input_width =   FLAGS.input_height
    if FLAGS.output_width is None: FLAGS.output_width = FLAGS.output_height
    FLAGS.out_name = '{}-{}x{}'.format(FLAGS.dataset, FLAGS.output_height,FLAGS.output_width)
    FLAGS.out_dir = os.path.join(FLAGS.out_dir, FLAGS.out_name)
    FLAGS.checkpoint_dir = os.path.join(FLAGS.out_dir, FLAGS.checkpoint_dir)
    FLAGS.sample_dir = os.path.join(FLAGS.out_dir, FLAGS.sample_dir)
    if not os.path.exists(FLAGS.checkpoint_dir): os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir): os.makedirs(FLAGS.sample_dir)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True
    with tf.Session(config=run_config) as sess:
        dcgan = DCGAN(
          sess,
          input_width=FLAGS.input_width,
          input_height=FLAGS.input_height,
          output_width=FLAGS.output_width,
          output_height=FLAGS.output_height,
          batch_size=FLAGS.batch_size,
          sample_num=FLAGS.batch_size,
          z_dim=FLAGS.z_dim,
          dataset_name=FLAGS.dataset,
          checkpoint_dir=FLAGS.checkpoint_dir,
          sample_dir=FLAGS.sample_dir,
          data_dir=FLAGS.data_dir,
          out_dir=FLAGS.out_dir,
          max_to_keep=FLAGS.max_to_keep)
        dcgan.train(FLAGS)
    
if __name__ == "__main__":
    tf.app.run()