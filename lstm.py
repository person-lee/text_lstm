
import tensorflow as tf

class LSTM(object):
    def __init__(self, batch_size, num_unroll_steps, embeddings, embedding_size, rnn_size, num_rnn_layers, num_classes, max_grad_norm, dropout = 1., l2_reg_lambda=0.0, adjust_weight=False,label_weight=[],is_training=True):
        # define input variable
        self.keep_prob = dropout
        self.batch_size = batch_size
        self.embeddings = embeddings
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.adjust_weight = adjust_weight
        self.label_weight = label_weight
        self.rnn_size = rnn_size
        self.num_rnn_layers = num_rnn_layers
        self.num_unroll_steps = num_unroll_steps
        self.l2_reg_lambda = l2_reg_lambda
        self.max_grad_norm = max_grad_norm

        self.input_data=tf.placeholder(tf.int32,[None,self.num_unroll_steps])
        self.target = tf.placeholder(tf.int64,[None])
        self.mask_x = tf.placeholder(tf.float32,[self.num_unroll_steps,None])

        #build LSTM network
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size, forget_bias=0.0, state_is_tuple=True)
        if self.keep_prob < 1:
            lstm_cell =  tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob = self.keep_prob
            )

        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.num_rnn_layers, state_is_tuple=True)

        self._initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)

        #embedding layer
        with tf.device("/cpu:0"),tf.name_scope("embedding_layer"):
            inputs=tf.nn.embedding_lookup(self.embeddings, self.input_data)

        if self.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, self.keep_prob)

        out_put=[]
        state=self._initial_state
        with tf.variable_scope("LSTM_layer"):
            for time_step in range(self.num_unroll_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state)=cell(inputs[:,time_step, :], state)
                out_put.append(cell_output)

        out_put=out_put * self.mask_x[:,:,None]

        with tf.name_scope("mean_pooling_layer"):
            out_put=tf.reduce_sum(out_put,0)/(tf.reduce_sum(self.mask_x,0)[:,None])

        with tf.name_scope("Softmax_layer_and_output"):
            softmax_w = tf.get_variable("softmax_w",[self.rnn_size, self.num_classes],dtype=tf.float32)
            softmax_b = tf.get_variable("softmax_b",[self.num_classes],dtype=tf.float32)
            self.logits = tf.matmul(out_put,softmax_w)+softmax_b

        with tf.name_scope("loss"):
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits+1e-10,self.target)
            self.cost = tf.reduce_mean(self.loss)

        with tf.name_scope("accuracy"):
            self.prediction = tf.argmax(self.logits,1)
            correct_prediction = tf.equal(self.prediction,self.target)
            self.correct_num=tf.reduce_sum(tf.cast(correct_prediction,tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name="accuracy")

        #add summary
        loss_summary = tf.scalar_summary("loss",self.cost)
        #add summary
        accuracy_summary=tf.scalar_summary("accuracy_summary",self.accuracy)

        if not is_training:
            return

        self.globle_step = tf.Variable(0,name="globle_step",trainable=False)
        self.lr = tf.Variable(0.0,trainable=False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                      self.max_grad_norm)


        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in zip(grads, tvars):
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        self.grad_summaries_merged = tf.merge_summary(grad_summaries)

        self.summary =tf.merge_summary([loss_summary,accuracy_summary,self.grad_summaries_merged])

        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        optimizer.apply_gradients(zip(grads, tvars))
        self.train_op=optimizer.apply_gradients(zip(grads, tvars))

        self.new_lr = tf.placeholder(tf.float32,shape=[],name="new_learning_rate")
        self._lr_update = tf.assign(self.lr,self.new_lr)

    def assign_new_lr(self,session,lr_value):
        session.run(self._lr_update,feed_dict={self.new_lr:lr_value})
