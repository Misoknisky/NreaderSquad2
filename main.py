import tensorflow as tf
import ujson as json
import numpy as np
from tqdm import tqdm
import os

from model.rnet import R_NET
from model.bidaf import BIDAF
from utils.util import get_record_parser, convert_tokens, evaluate, get_batch_dataset, get_dataset

def select_model(str_model):
    train_model={"rnet":R_NET,"bidaf":BIDAF}
    if str_model in train_model.keys():
        return train_model[str_model]
    return train_model["rnet"]
def train(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.train_eval_file, "r") as fh:
        train_eval_file = json.load(fh)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    with open(config.dev_meta, "r") as fh:
        meta = json.load(fh)

    dev_total = meta["total"]
    print("Building model...")
    train_model=select_model(config.model)
    parser = get_record_parser(config)
    train_dataset = get_batch_dataset(config.train_record_file,parser,config)
    dev_dataset = get_dataset(config.dev_record_file, parser, config)
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
    train_iterator = train_dataset.make_one_shot_iterator()
    dev_iterator = dev_dataset.make_one_shot_iterator()

    model = train_model(config, iterator, word_mat, char_mat)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    loss_save = 100.0
    patience = 0
    max_metrics=-1
    lr = config.init_lr

    with tf.Session(config=sess_config) as sess:
        writer = tf.summary.FileWriter(config.log_dir)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        train_handle = sess.run(train_iterator.string_handle())
        dev_handle = sess.run(dev_iterator.string_handle())
        sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))
        sess.run(tf.assign(model.lr, tf.constant(lr, dtype=tf.float32)))

        for _ in tqdm(range(1, config.num_steps + 1)):
            global_step = sess.run(model.global_step) + 1
            loss,acc,train_op = sess.run([model.loss,model.acc,model.train_op], feed_dict={handle: train_handle})
            if global_step % config.period == 0:
                loss_sum = tf.Summary(value=[tf.Summary.Value(tag="model/loss", simple_value=loss), ])
                writer.add_summary(loss_sum, global_step)
                print("train step {} loss is {} acc is {}".format(global_step,loss,acc))
            if global_step % config.checkpoint == 0:
                sess.run(tf.assign(model.is_train,tf.constant(False, dtype=tf.bool)))
                _, summ = evaluate_batch(model, config.val_num_batches, train_eval_file, sess, "train", handle, train_handle)
                for s in summ:
                    writer.add_summary(s, global_step)
                metrics, summ = evaluate_batch(model, dev_total // config.batch_size + 1, dev_eval_file, sess, "dev", handle, dev_handle)
                sess.run(tf.assign(model.is_train,tf.constant(True, dtype=tf.bool)))
                dev_loss = metrics["loss"]
                if dev_loss < loss_save:
                    loss_save = dev_loss
                    patience = 0
                else:
                    patience += 1
                if patience >= config.patience:
                    lr /= 2.0
                    loss_save = dev_loss
                    patience = 0
                sess.run(tf.assign(model.lr, tf.constant(lr, dtype=tf.float32)))
                for s in summ:
                    writer.add_summary(s, global_step)
                writer.flush()
                filename = os.path.join(config.save_dir, "model_{}.ckpt".format(global_step))
                t_metrics=(metrics["f1"] + metrics["exact_match"]) / 2.0
                print("dev loss is {} f1_score {} exact_match_score {} metrics_score is {} acc {}".format(dev_loss,metrics["f1"],metrics["exact_match"],t_metrics,metrics["acc"]))
                if t_metrics > max_metrics:
                    max_metrics=t_metrics
                    saver.save(sess, filename)
                    print("save model path {}".format(filename))


def evaluate_batch(model, num_batches, eval_file, sess, data_type, handle, str_handle):
    answer_dict = {}
    losses = []
    accuracy=[]
    for _ in tqdm(range(1, num_batches + 1)):
        qa_id, loss, yp1, yp2,impossible,acc= sess.run([model.qa_id, model.loss, model.yp1, model.yp2,model.pre_impossible,model.acc],feed_dict={handle: str_handle})
        answer_dict_, _ = convert_tokens(eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist(),impossible)
        answer_dict.update(answer_dict_)
        losses.append(loss)
        accuracy.append(acc)
    t_acc=np.mean(accuracy)
    loss = np.mean(losses)
    metrics = evaluate(eval_file, answer_dict)
    metrics["loss"] = loss
    metrics["acc"]=t_acc
    loss_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/loss".format(data_type), simple_value=metrics["loss"]), ])
    f1_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/f1".format(data_type), simple_value=metrics["f1"]), ])
    em_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/em".format(data_type), simple_value=metrics["exact_match"]), ])
    acc_sum=tf.Summary(value=[tf.Summary.Value(
        tag="{}/acc".format(data_type), simple_value=metrics["acc"]), ])
    return metrics, [loss_sum, f1_sum, em_sum,acc_sum]


def test(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.test_eval_file, "r") as fh:
        eval_file = json.load(fh)
    with open(config.test_meta, "r") as fh:
        meta = json.load(fh)

    total = meta["total"]

    print("Loading model...")
    train_model=select_model(config.model)
    test_batch = get_dataset(config.test_record_file, get_record_parser(
        config, is_test=True), config).make_one_shot_iterator()

    model = train_model(config, test_batch, word_mat, char_mat, trainable=False)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
        sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))
        losses = []
        answer_dict = {}
        remapped_dict = {}
        for step in tqdm(range(total // config.batch_size + 1)):
            qa_id, loss, yp1, yp2,possible= sess.run([model.qa_id, model.loss, model.yp1, model.yp2,model.pre_impossible])
            answer_dict_, remapped_dict_ = convert_tokens(eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist(),possible)
            answer_dict.update(answer_dict_)
            remapped_dict.update(remapped_dict_)
            losses.append(loss)
        loss = np.mean(losses)
        metrics = evaluate(eval_file, answer_dict)
        with open(config.answer_file, "w") as fh:
            json.dump(remapped_dict, fh)
        print("Exact Match: {}, F1: {}".format(metrics['exact_match']))
