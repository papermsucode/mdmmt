import os
import numpy as np


import tensorflow.compat.v1 as tf
import vggish.vggish_input as vggish_input
import vggish.vggish_params as vggish_params
import vggish.vggish_postprocess as vggish_postprocess
import vggish.vggish_slim as vggish_slim

class VGGish:
    def __init__(self, ckpt_path, per_batch_size):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.per_batch_size = per_batch_size
        # sys.path.append('/home/wx762845/src/models-master/research/audioset/vggish/') # TODO

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.2
        self.manager=tf.Graph().as_default()
        self.manager.__enter__()
        self.sess =  tf.Session(config = config)

        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(self.sess, ckpt_path) # /home/wx762845/src/models-master/research/audioset/vggish/vggish_model.ckpt
        self.features_tensor = self.sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        self.embedding_tensor = self.sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME)
 
    def __call__(self, frames):
        mel_features = np.stack(frames)

        [outputs] = self.sess.run([self.embedding_tensor],
                            feed_dict={self.features_tensor: mel_features})
        relu = lambda x: np.maximum(x, 0)
        embs = relu(outputs)  

        return embs
