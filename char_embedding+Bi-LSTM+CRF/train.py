#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config

print("Import successfully")


# In[ ]:


import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
 # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print('----------START GPU----------')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            print('----------END GPU----------')
    except RuntimeError as e:
 # Memory growth must be set before GPUs have been initialized
        print(e)


# In[ ]:


def main():
    # create instance of config
    print("Start to Train")
    config = Config()

    # build model
    model = NERModel(config)
    model.build()
    # model.restore_session("results/crf/model.weights/") # optional, restore weights
    # model.reinitialize_weights("proj")

    # create datasets
    dev   = CoNLLDataset(config.filename_dev, config.processing_word,
                         config.processing_tag, config.max_iter)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)

    # train model
    model.train(train, dev)

if __name__ == "__main__":
    main()

