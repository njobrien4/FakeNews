#! /usr/bin/env python

import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.nan)
import os
import data_helpers
from tensorflow.contrib import learn
import csv
from sklearn import metrics
import interpret
#import yaml


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))

cfg = {'word_embeddings': {'default': 'word2vec', 'word2vec': {'path': '/om/user/njobrien/FakeNews/cnn-text-classification-tf (body)/data/word_embeddings/GoogleNews-vectors-negative300.bin', 'dimension': 300, 'binary': True}, 'glove': {'path': '../../data/glove.6B.100d.txt', 'dimension': 100, 'length': 400000}}, 'datasets': {'default': '20newsgroup', 'mrpolarity': {'positive_data_file': {'path': 'data/rt-polaritydata/rt-polarity.pos', 'info': 'Data source for the positive data'}, 'negative_data_file': {'path': 'data/rt-polaritydata/rt-polarity.neg', 'info': 'Data source for the negative data'}}, '20newsgroup': {'categories': ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian'], 'shuffle': True, 'random_state': 42}, 'localdata': {'container_path': '../../data/input/SentenceCorpus', 'categories': None, 'shuffle': True, 'random_state': 42}}}

# Parameters
# ==================================================

# Data Parameters

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")


tf.flags.DEFINE_string("positive_data_file", "/om/user/njobrien/FakeNews/cnn-text-classification-tf (body)/data/news-data/real_bodies.txt", "Data source for the real data.")
tf.flags.DEFINE_string("negative_data_file", "/om/user/njobrien/FakeNews/cnn-text-classification-tf (body)/data/news-data/fake_bodies.txt", "Data source for the fake data.")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

datasets = None

# CHANGE THIS: Load data. Load your own data here
dataset_name = cfg["datasets"]["default"]
if FLAGS.eval_train:
##    if dataset_name == "mrpolarity":
##        datasets = data_helpers.get_datasets_mrpolarity(cfg["datasets"][dataset_name]["positive_data_file"]["path"],
##                                             cfg["datasets"][dataset_name]["negative_data_file"]["path"])
##    elif dataset_name == "20newsgroup":
##        datasets = data_helpers.get_datasets_20newsgroup(subset="test",
##                                              categories=cfg["datasets"][dataset_name]["categories"],
##                                              shuffle=cfg["datasets"][dataset_name]["shuffle"],
##                                              random_state=cfg["datasets"][dataset_name]["random_state"])
    
    x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    y_test = np.argmax(y_test, axis=1)
    print("Total number of test examples: {}".format(len(y_test)))
else:
    if dataset_name == "mrpolarity":
        datasets = {"target_names": ['positive_examples', 'negative_examples']}
        x_raw = ["a masterpiece four years in the making", "everything is off."]
        y_test = [1, 0]
    else:
        datasets = {"target_names": ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']}
        x_raw = ["The number of reported cases of gonorrhea in Colorado increased",
                 "I am in the market for a 24-bit graphics card for a PC"]
        y_test = [2, 1]

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        scores = graph.get_operation_by_name("output/scores").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        conv_mp3 = graph.get_operation_by_name("conv-maxpool-3/conv").outputs[0]
        relu_mp3 = graph.get_operation_by_name("conv-maxpool-3/relu").outputs[0]
        before_predictions=graph.get_operation_by_name("W").outputs[0]
        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)
        #output_w = graph.get_operation_by_name("output/W").outputs[0]
        b = graph.get_operation_by_name("output/b").outputs[0]
        pool_mp3 = graph.get_operation_by_name("conv-maxpool-3/pool").outputs[0]
        conv_lensequence = graph.get_operation_by_name("conv-maxpool-3/conv").outputs[0]
        h_drop = graph.get_operation_by_name("dropout/dropout/mul").outputs[0]
        # Collect the predictions here
        all_predictions = []
        all_probabilities = None
        
        best_trigrams ={}
        for x_test_batch in batches:
            batch_predictions_scores = sess.run([predictions, scores,conv_mp3,before_predictions,b,pool_mp3,h_drop,conv_lensequence,relu_mp3], {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_vars=tf.trainable_variables()
#             print(all_vars, "is all vars")
#             print(b, "is b")



#             #conv_mp3 = graph.get_operation_by_name("conv-maxpool-3/conv")
#             #a=tf.Print(conv_mp3.outputs[0],[conv_mp3.outputs[0]],message="this is conv outputs")
#             #b   = tf.add(a, a)
#             #b.eval()           
# #for var in all_vars:
#                 #W=sess.run(var)
#                 #print(W, "is",var
            
        
#             #print(np.sum(batch_predictions_scores[3],axis=0))


#             print(batch_predictions_scores[4], "is b")
#             #print(batch_predictions_scores[6],batch_predictions_scores[6].shape, "is output w")
#             #print(batch_predictions_scores[5].squeeze(), batch_predictions_scores[5].shape, "is pool")
            # print("pool is x * output w + b")
            print(batch_predictions_scores[3].shape, "is W")
            conv=batch_predictions_scores[7]
            # print(conv[0][:13],conv.shape, "is conv[0]")
            # print(batch_predictions_scores[6], "is h_drop")
            all_predictions = np.concatenate([all_predictions, batch_predictions_scores[0]])
            probabilities = softmax(batch_predictions_scores[1])
            xW=np.matmul(batch_predictions_scores[6],batch_predictions_scores[3])
            relu_result = batch_predictions_scores[8]
            # print(relu_result[:13],batch_predictions_scores[8].shape, "is relu[:13]")
            pool_post_relu = batch_predictions_scores[5]
            # print(pool_post_relu, "is pool after relu")
            # print(xW,"is xW")
            b_result=batch_predictions_scores[4]
            # print(xW+b_result, "is xw + b")
            # print(batch_predictions_scores[1], "is plain scores")
            # print(softmax(batch_predictions_scores[1]), "is softmax scores")
            # print(all_predictions, "is all_predictions")
            # print(probabilities, " is scores")
            # #conv_mp3 = batch_predictions_scores[2]
            # print(conv_mp3.shape, "is shape")
            sums=np.sum(conv_mp3,axis=-1)
           # print(sums[0][:20],sums[1][:20])
           # print(conv_mp3, "is convmp3")
            
            best_trigrams = interpret.interpret_many(x_raw,relu_result, pool_post_relu, best_trigrams)
            print (len(best_trigrams[1]), "is len best_trigrams[1]")
            if all_probabilities is not None:
                all_probabilities = np.concatenate([all_probabilities, probabilities])
            else:
                all_probabilities = probabilities

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    #for each thing in all predictions, if its equal to y test you wanna print x_raw

    wrong = [(x_raw[i],y_test[i]) for i in range(len(y_test)) if all_predictions[i]!=y_test[i]]
    #print("wrongs", wrong)
    with open("false_pos.txt", 'w') as false_pos, open('false_neg.txt', 'w') as false_neg:
        for headline, num in wrong:
            if num==0:
                # should be fake, but was real
                false_pos.write(headline+"\n")
            else:
                #should be real, but was fake
                false_neg.write(headline+"\n")
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
   # print(metrics.classification_report(y_test, all_predictions, target_names=datasets['target_names']))
    print(metrics.confusion_matrix(y_test, all_predictions))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw),
                                              [int(prediction) for prediction in all_predictions],
                                              [ "{}".format(probability) for probability in all_probabilities]))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
#print(best_trigrams)
with open('best_trigrams.txt', 'w') as f: 
    for k in best_trigrams.keys():
        list_o_lists=best_trigrams[k]
        best_trigrams_for_k=[]
        for li in list_o_lists:
            if len(li)>0:
                try:
                    trigram = ' '.join(li)
                    print("it worked")
                    print("trigram: ", trigram)
                except: 
                    trigram = ' '.join(li[0])
                    print("it didnt,",trigram)
                best_trigrams_for_k.append(trigram)
        #print(np.array(list_o_lists[3]).squeeze(), "is list")
        #list_o_strings = [' '.join(list(np.array(lil_list).squeeze())) for lil_list in list_o_lists]
        f.write("i: "+str(k)+'\n')
        f.write("trigrams: ")
        for trigram in best_trigrams_for_k:
            f.write(trigram+",")
        f.write('\n')





