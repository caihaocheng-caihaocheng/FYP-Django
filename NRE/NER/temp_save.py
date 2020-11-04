import tensorflow as tf
import numpy as np
import os
import subprocess

from .data_helpers import *
from .utils import *
from .configure import FLAGS

text_path = os.path.join("runs/1601652948/checkpoints/../vocab")
print(text_path)
print("====================================================")
text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(text_path)
checkpoint_file = tf.train.latest_checkpoint("runs/1601652948/checkpoints/")

def predict_RE(input_sentence):
    with tf.device('/cpu:0'):
        input_sentence_list = []
        input_sentence_list.append(input_sentence)
        input_sentence_transform = np.array(list(text_vocab_processor.transform(input_sentence_list)))
        input_sentence_transform = np.array(input_sentence_transform)

        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)
            session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)

                # Get the placeholders from the graph by name
                input_text = graph.get_operation_by_name("input_text").outputs[0]
                # input_y = graph.get_operation_by_name("input_y").outputs[0]
                emb_dropout_keep_prob = graph.get_operation_by_name("emb_dropout_keep_prob").outputs[0]
                rnn_dropout_keep_prob = graph.get_operation_by_name("rnn_dropout_keep_prob").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                    # Tensors we want to evaluate
                predictions = graph.get_operation_by_name("output/predictions").outputs[0]

                pred_test = sess.run(predictions, {input_text:input_sentence_transform[0].reshape(1,90), emb_dropout_keep_prob:1.0, rnn_dropout_keep_prob:1.0, dropout_keep_prob:1.0})
                prediction = utils.label2class[pred_test[0]]
                print("pred_test:", pred_test, prediction)
                input_sentence_list.clear()



def main(_):
    sentence = "The most common e11 audits e12 were about e21 waste e22 and recycling."
    predict_RE(sentence)


if __name__ == "__main__":
    tf.app.run()
