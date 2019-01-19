#coding=utf-8
'''
Created on Oct 25, 2018

@author: liuyongkang
'''
import tensorflow as tf
from utils.func import cudnn_gru, native_gru, summ, dropout, ptr_net
from utils.basic_rnn import rnn
class AttentionFlowMatchLayer(object):
    """
    Implements the Attention Flow layer,
    which computes Context-to-question Attention and question-to-context Attention
    """
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def match(self, passage_encodes, question_encodes):
        """
        Match the passage_encodes with question_encodes using Attention Flow Match algorithm
        """
        with tf.variable_scope('bidaf'):
            sim_matrix = tf.matmul(passage_encodes, question_encodes, transpose_b=True)
            context2question_attn = tf.matmul(tf.nn.softmax(sim_matrix, -1), question_encodes)
            b = tf.nn.softmax(tf.expand_dims(tf.reduce_max(sim_matrix, 2), 1), -1)
            question2context_attn = tf.tile(tf.matmul(b, passage_encodes),
                                         [1, tf.shape(passage_encodes)[1], 1])
            concat_outputs = tf.concat([passage_encodes, context2question_attn,
                                        passage_encodes * context2question_attn,
                                        passage_encodes * question2context_attn], -1)
            return concat_outputs, None
class SelfAttLayer(object):

    def __init__(self, hidden):
        self.hidden = hidden

    def bi_linear_att(self, inputs, memory):
        with tf.variable_scope("self_attention"):
            i_dim = inputs.get_shape().as_list()[-1]
            flat_inputs = tf.reshape(inputs, [-1, i_dim])
            m_dim = inputs.get_shape().as_list()[-1]
            weight = tf.get_variable("W", [i_dim, m_dim])
            shape = tf.shape(inputs)
            out_shape = [shape[idx] for idx in range(len(inputs.get_shape().as_list()) - 1)] + [m_dim]
            result_input = tf.reshape(tf.matmul(flat_inputs, weight), out_shape)
            outputs = tf.nn.relu(tf.matmul(result_input, tf.transpose(memory, [0, 2, 1]))) / (self.hidden ** 0.5)
            logits = tf.nn.softmax(outputs)
            outputs = tf.matmul(logits, memory)
            return tf.concat([inputs, outputs], axis=2)
class BIDAF(object):
    def highway(self,x,y,units, scope):
        with tf.variable_scope(scope):
            if x != None:
                x = tf.layers.dense(x, units, activation=tf.nn.relu)
            if y != None:
                y = tf.layers.dense(y, units, activation=tf.nn.relu)
            gate = tf.layers.dense(x, units, activation=tf.nn.sigmoid)
            if y !=None:
                output = y*gate + (1-gate)*x
            else:
                output = gate * x
        return output
    def __init__(self, config, batch, word_mat=None, char_mat=None, trainable=True, opt=True):
        self.config = config
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,initializer=tf.constant_initializer(0), trainable=False)
        self.c, self.q, self.ch, self.qh, self.y1, self.y2, self.qa_id,self.impossible= batch.get_next()
        self.is_train = tf.get_variable("is_train", shape=[], dtype=tf.bool, trainable=False)
        self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(word_mat, dtype=tf.float32), trainable=False)
        self.char_mat = tf.get_variable("char_mat", initializer=tf.constant(char_mat, dtype=tf.float32))
        self.p_word_mat=tf.get_variable("p_word_mat", initializer=tf.constant(word_mat, dtype=tf.float32), trainable=False)
        self.p_char_mat=tf.get_variable("p_char_mat", initializer=tf.constant(char_mat, dtype=tf.float32))
        self.c_mask = tf.cast(self.c, tf.bool)
        self.q_mask = tf.cast(self.q, tf.bool)
        self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)
        if opt:
            N, CL = config.batch_size, config.char_limit
            self.c_maxlen = tf.reduce_max(self.c_len)
            self.q_maxlen = tf.reduce_max(self.q_len)
            self.c = tf.slice(self.c, [0, 0], [N, self.c_maxlen])
            self.q = tf.slice(self.q, [0, 0], [N, self.q_maxlen])
            self.c_mask = tf.slice(self.c_mask, [0, 0], [N, self.c_maxlen])
            self.q_mask = tf.slice(self.q_mask, [0, 0], [N, self.q_maxlen])
            self.ch = tf.slice(self.ch, [0, 0, 0], [N, self.c_maxlen, CL])
            self.qh = tf.slice(self.qh, [0, 0, 0], [N, self.q_maxlen, CL])
            self.y1 = tf.slice(self.y1, [0, 0], [N, self.c_maxlen])
            self.y2 = tf.slice(self.y2, [0, 0], [N, self.c_maxlen])
            self.impossible=tf.slice(self.impossible, [0, 0], [N, 2])
        else:
            self.c_maxlen, self.q_maxlen = config.para_limit, config.ques_limit
        self.ch_len = tf.reshape(tf.reduce_sum(tf.cast(tf.cast(self.ch, tf.bool), tf.int32), axis=2), [-1])
        self.qh_len = tf.reshape(tf.reduce_sum(tf.cast(tf.cast(self.qh, tf.bool), tf.int32), axis=2), [-1])
        self.ready()
        if trainable:
            self.lr = tf.get_variable("lr", shape=[], dtype=tf.float32, trainable=False)
            self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.lr, epsilon=1e-6)
            grads = self.opt.compute_gradients(self.loss)
            gradients, variables = zip(*grads)
            capped_grads, _ = tf.clip_by_global_norm(gradients, config.grad_clip)
            self.train_op = self.opt.apply_gradients(zip(capped_grads, variables), global_step=self.global_step)

    def ready(self):
        config = self.config
        N, PL, QL, CL, d, dc, dg = config.batch_size, self.c_maxlen, self.q_maxlen, config.char_limit, config.hidden, config.char_dim, config.char_hidden
        dc = config.glove_dim if config.pretrained_char else config.char_dim
        gru = cudnn_gru if config.use_cudnn else native_gru
        with tf.variable_scope("emb"):
            with tf.variable_scope("char"):
                ch_emb = tf.reshape(tf.nn.embedding_lookup(self.char_mat, self.ch), [N * PL, CL, dc])
                qh_emb = tf.reshape(tf.nn.embedding_lookup(self.char_mat, self.qh), [N * QL, CL, dc])
                cell_fw = tf.contrib.rnn.GRUCell(dg)
                cell_bw = tf.contrib.rnn.GRUCell(dg)
                _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, ch_emb, self.ch_len, dtype=tf.float32)
                ch_emb = tf.concat([state_fw, state_bw], axis=1)
                _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, qh_emb, self.qh_len, dtype=tf.float32)
                qh_emb = tf.concat([state_fw, state_bw], axis=1)
                qh_emb = tf.reshape(qh_emb, [N, QL, 2 * dg])
                ch_emb = tf.reshape(ch_emb, [N, PL, 2 * dg])
            with tf.name_scope("word"):
                c_emb = tf.nn.embedding_lookup(self.word_mat, self.c)
                q_emb = tf.nn.embedding_lookup(self.word_mat, self.q)
            c_emb = tf.concat([c_emb, ch_emb], axis=2)
            q_emb = tf.concat([q_emb, qh_emb], axis=2)
        with tf.variable_scope("extract_info"):
            dim=2*dg + config.glove_dim
            weight=tf.get_variable(name="extract_q_info", shape=[dim,dim])
            att=tf.nn.softmax(tf.einsum("bpd,bqd->bpq",tf.einsum("bpd,dt->bpt",c_emb,weight),q_emb),axis=-1)#bpq
            Dc=tf.reduce_sum(tf.einsum("bpq,bqd->bpqd",att,q_emb),axis=2)#bpd
            att_=tf.transpose(att, perm=[0,2,1])
            Qc=tf.reduce_sum(tf.einsum("bqp,bpd->bqpd",att_,c_emb),axis=2)#bqd
            c_emb=tf.concat(values=[c_emb,Dc],axis=-1)
            q_emb=tf.concat(values=[q_emb,Qc],axis=-1)
        with tf.variable_scope("passage_encoding"):
            c,_=rnn('bi-gru',c_emb, self.c_len,d)
            c=self.highway(c_emb,c, units=2 * d, scope="gate_context")
        with tf.variable_scope("question_encoding"):
            q,_=rnn('bi-gru',q_emb, self.q_len,d)
            q=self.highway(q_emb,q, units=2 * d, scope="gate_question")
        with tf.variable_scope("gate"):
            c=dropout(c, keep_prob=config.keep_prob, is_train=self.is_train)
            q=dropout(q, keep_prob=config.keep_prob, is_train=self.is_train)
        with tf.variable_scope("impossible"):
            weight=tf.get_variable(name="weight_im", shape=[2*d])
            att_q=tf.nn.softmax(tf.einsum("bqd,d->bq",q,weight),axis=-1)
            q_enc=tf.reduce_sum(tf.einsum("bqd,bq->bqd",q,att_q),axis=1)#bd
            weight_2=tf.get_variable(name="weight_2", shape=[2*d,2*d,2*d])
            h=tf.reduce_max(tf.einsum("bptd,bd->bpt",tf.einsum("bpd,dtm->bptm",c,weight_2),q_enc),axis=1)#bd
            weight_3=tf.get_variable(name="weight_3",shape=[2*d,2])
            score=tf.einsum("bd,dt->bt",h,weight_3)
        with tf.variable_scope("match"):
            match_layer = AttentionFlowMatchLayer(d)
            match_p_encodes, _ = match_layer.match(c,q)
            match_p_encodes=dropout(match_p_encodes, keep_prob=config.keep_prob, is_train=self.is_train)
        with tf.variable_scope("fusion"):
            fuse_info,_=rnn('bi-gru',match_p_encodes, self.c_len,d)
            fuse_info=self.highway(fuse_info, None, units=2*d, scope="gate_fusion")
        with tf.variable_scope("attention"):
            attention_layer = SelfAttLayer(d)
            self_att = attention_layer.bi_linear_att(fuse_info,fuse_info)
            para_p_encodes, _ = rnn('bi-gru', self_att,self.c_len, d)
        with tf.variable_scope("pointer"):
            init = summ(q[:, :, -2 * d:], d, mask=self.q_mask,keep_prob=config.ptr_keep_prob, is_train=self.is_train)
            pointer = ptr_net(batch=N, hidden=init.get_shape().as_list()[-1], keep_prob=config.ptr_keep_prob, is_train=self.is_train)
            logits1, logits2 = pointer(init, para_p_encodes, d, self.c_mask)#bp
        with tf.variable_scope("predict"):
            outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),tf.expand_dims(tf.nn.softmax(logits2), axis=1))
            self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
            self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
            self.pre_impossible=tf.argmax(score,axis=-1)
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits1, labels=tf.stop_gradient(self.y1))
            losses2 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits2, labels=tf.stop_gradient(self.y2))
            losses3=tf.nn.softmax_cross_entropy_with_logits_v2(logits=score, labels=tf.stop_gradient(self.impossible))
            self.loss = tf.reduce_mean(losses + losses2)+0.5 * tf.reduce_mean(losses3)
        with tf.variable_scope("accuracy"):
            self.acc=tf.reduce_mean(tf.cast(tf.equal(self.pre_impossible,tf.argmax(self.impossible,axis=-1)),tf.float32))
    def get_loss(self):
        return self.loss
    def get_global_step(self):
        return self.global_step