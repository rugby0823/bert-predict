# import os
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import tokenization
import modeling
from time import time


# two texts pre-process
def text_process(text1, text2, tokenizer, max_seq_length):
    tokens_1 = tokenizer.tokenize(text1)
    tokens_2 = tokenizer.tokenize(text2)

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_1:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_2:
        for token in tokens_2:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    return input_ids, input_mask, segment_ids


# restore finetuned model
def init(max_sequence_length, bert_config_file, model_path, vocab_file):
    sess = tf.Session()
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)

    input_ids = tf.placeholder(tf.int32, shape=[None, max_sequence_length], name='input_ids')
    input_mask = tf.placeholder(tf.int32, shape=[None, max_sequence_length], name='input_mask')
    segment_ids = tf.placeholder(tf.int32, shape=[None, max_sequence_length], name='segment_ids')

    with sess.as_default():
        model = modeling.BertModel(
            config=bert_config,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False)

        output_layer = model.get_pooled_output()

        hidden_size = output_layer.shape[-1].value

        output_weights = tf.get_variable(
            "output_weights", [2, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [2], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):
            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            probabilities = tf.nn.softmax(logits, axis=-1)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_path)

    return sess, tokenizer


def predict(sess, input_ids, input_mask, segment_ids):
    input_ids_tensor = sess.graph.get_tensor_by_name('input_ids:0')
    input_mask_tensor = sess.graph.get_tensor_by_name('input_mask:0')
    segment_ids_tensor = sess.graph.get_tensor_by_name('segment_ids:0')
    output_tensor = sess.graph.get_tensor_by_name('loss/Softmax:0')

    fd = {input_ids_tensor: [input_ids], input_mask_tensor: [input_mask], segment_ids_tensor: [segment_ids]}
    output_result = sess.run([output_tensor], feed_dict=fd)

    return output_result


if __name__ == "__main__":
    # ######filepath and paramaters#############
    # the pre-trained files supplied by google, here gives vocab.txt and bert_config.json for the chinese pre-trained
    # model, you can get them from https://github.com/google-research/bert
    vocab_file = 'model/chinese_L-12_H-768_A-12/vocab.txt'
    bert_config_file = 'model/chinese_L-12_H-768_A-12/bert_config.json'

    # the model path, you need to put your trained model in the path
    model_path = 'model/model.ckpt'
    max_sequence_length = 128
    # #########################################

    # the two strings you use to predict
    text1 = '我爱你'
    text2 = '我恨你'

    # init model
    t1 = time()
    sess, tokenizer = init(max_sequence_length, bert_config_file, model_path, vocab_file)
    print("init time: %.4f s" % (time() - t1))

    # predict
    t2 = time()
    input_ids, input_mask, segment_ids = text_process(text1, text2, tokenizer, max_sequence_length)
    result = predict(sess, input_ids, input_mask, segment_ids)
    print(result[0][0])
    print("predict time: %.4f s" % (time() - t2))

