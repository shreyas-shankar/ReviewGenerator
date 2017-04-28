import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from DataReader import CharReader
from DataReader import WordReader

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("save_model_path", None, "directory to output the model")
flags.DEFINE_string("data_path", "negative_reviews.txt", "The path point to the training and testing data")
flags.DEFINE_integer("ckpt", 1, "Checkpoint after this many steps (default: 100)")
flags.DEFINE_string("model", "word_model", "choose the model")
#flags.DEFINE_string("model", "char_model", "choose the model")
out_path = "review_out.txt"


FLAGS = flags.FLAGS

class LanguageModel(object):

    def __init__(self, config, is_training=True):
        self.config = config
        self.input_x_seq = tf.placeholder(tf.int32, [self.config.batch_size, self.config.sequence_size], "input_x_seq")
        self.output_y_seq = tf.placeholder(tf.int32, [self.config.batch_size, self.config.sequence_size], "output_y_seq")

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [self.config.vocab_size, self.config.hidden_size])
            embedding_looked_up = tf.nn.embedding_lookup(embedding, self.input_x_seq)
            inputs = tf.split(embedding_looked_up, self.config.sequence_size, 1)
            input_tensors_list = [tf.squeeze(i, [1]) for i in inputs]

        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size, state_is_tuple=True,reuse=tf.get_variable_scope().reuse)

        single_cell = lstm_cell

        #if is_training and self.config.keep_prob < 1:
        #    def single_cell():
        #        return tf.nn.rnn_cell.DropoutWrapper(lstm_cell(), output_keep_prob=self.config.keep_prob)

        self.cells =  tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(self.config.num_layers)])
        self._initial_state = self.cells.zero_state(self.config.batch_size, tf.float32)

        #logits for softmax
        hidden_layer_output, last_state = tf.contrib.rnn.static_rnn(self.cells, input_tensors_list, initial_state=self._initial_state)
        hidden_layer_output = tf.reshape(tf.concat(hidden_layer_output, 1), [-1, self.config.hidden_size])
        self._logits = tf.nn.xw_plus_b(hidden_layer_output, tf.get_variable("softmax_w", [self.config.hidden_size, self.config.vocab_size]), tf.get_variable("softmax_b", [self.config.vocab_size]))
        self._predictions = tf.nn.softmax(self.logits)

        #loss function
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([self._logits], [tf.reshape(self.output_y_seq, [-1])], [tf.ones([self.config.batch_size * self.config.sequence_size])], self.config.vocab_size)
        self._cost = tf.div(tf.reduce_sum(loss), self.config.batch_size)

        self._final_state = last_state

    def GradientDescentOptimize(self):
        self._learning_rate = tf.Variable(0.0, trainable=False)

        training_vars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, training_vars),self.config.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.run_gradient_descent_optimizer = optimizer.apply_gradients(zip(grads, training_vars))


    def assign_learning_rate(self, session, lr_value):
        session.run(tf.assign(self.learning_rate, lr_value))

    @property
    def input_seq(self):
        return self.input_x_seq

    @property
    def target_seq(self):
        return self.output_y_seq

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def gradient_desc_training_op(self):
        return self.run_gradient_descent_optimizer

    @property
    def prediction_softmax(self):
        return self._predictions

    @property
    def logits(self):
        return self._logits

def get_config():
    if FLAGS.model == "word_model":
        return WordModel()
    elif FLAGS.model == "word_model_dp":
        return WordModelWithDropuout()
    elif FLAGS.model == "char_model":
        return CharModel()
    elif FLAGS.model == "char_model_dp":
        return CharModelWithDropout()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)

def get_reader():
    if FLAGS.model == "word_model" or FLAGS.model == "word_model_dp":
        return WordReader(FLAGS.data_path)
    elif FLAGS.model == "char_model" or FLAGS.model == "char_model_dp":
        return CharReader(FLAGS.data_path)
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)

def main(unused_args):

    #reader = DataReader(FLAGS.data_path)
    reader = get_reader()

    with tf.Graph().as_default(), tf.Session() as session:
        config = get_config()
        config.vocab_size = reader.vocab_size

        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        with tf.variable_scope("LanguageModel", reuse=None, initializer=initializer):
          training_model = LanguageModel(config)
          training_model.GradientDescentOptimize()
        with tf.variable_scope("LanguageModel", reuse=True, initializer=initializer):
          eval_config = get_config()
          eval_config.vocab_size = reader.vocab_size
          eval_config.batch_size = 1
          eval_config.num_steps = 1
          prediction_model = LanguageModel(eval_config)

          tf.global_variables_initializer().run()

        for epoch in range(config.max_max_epoch):
            accumulated_costs = 0.0
            accumulated_seq_count = 0
            current_state = session.run(training_model.initial_state)

            learning_rate_decay = config.lr_decay ** max(epoch - config.max_epoch, 0.0)
            training_model.assign_learning_rate(session, config.learning_rate * learning_rate_decay)

            lowest_perplexity = 2000

            for seq_counter, (x, y) in enumerate(reader.generateXYPairs(reader.get_training_data(), training_model.config.batch_size, training_model.config.sequence_size)):
                feed_dict = {training_model.input_x_seq: x, training_model.output_y_seq: y, training_model.initial_state: current_state}

                cost, current_state, _ = session.run([training_model.cost, training_model.final_state, training_model.gradient_desc_training_op], feed_dict)
                accumulated_costs += cost
                accumulated_seq_count += training_model.config.sequence_size
                perplexity =  np.exp(accumulated_costs / accumulated_seq_count)

                if  seq_counter != 0 and seq_counter % tf.flags.FLAGS.ckpt == 0:
                  print("Epoch %d, Perplexity: %.3f" % (epoch, perplexity))

                  with open(out_path,'a') as out:
                    out.write(str("\nEpoch %d, Perplexity: %.3f" % (epoch, perplexity)))

                  if perplexity < lowest_perplexity:
                    lowest_perplexity = perplexity
                    #get_prediction(prediction_model, reader, session, 500, ['T','h','e',' '])
                    # if FLAGS.model == "char_model":
                    #     get_prediction(prediction_model, reader, session, 500, ['T','h','e',' '])
                    # elif FLAGS.model == "word_model":
                    #     get_prediction(prediction_model, reader, session, 50, [''])
                  if FLAGS.model == "char_model":
                    get_prediction(prediction_model, reader, session, 500, ['T','h','e',' '])
                  elif FLAGS.model == "word_model":
                    get_prediction(prediction_model, reader, session, 50, [''])


    session.close()


def get_prediction(model, reader, session, total_tokens, output_tokens = ['']):

  state = session.run(model.cells.zero_state(1, tf.float32))

  #print total_tokens

  for token_count in range(total_tokens):
      next_token = output_tokens[token_count]
      input = np.full((model.config.batch_size, model.config.sequence_size), reader.token_to_id[next_token], dtype=np.int32)
      feed = {model.input_x_seq: input, model.initial_state:state}
      [prediction_softmax, state] =  session.run([model.prediction_softmax, model.final_state], feed)

      if (len(output_tokens) -1) <= token_count:
          accumulated_sum = np.cumsum(prediction_softmax[0])
          currentTokenId = (int(np.searchsorted(accumulated_sum, np.random.rand(1))))
          if currentTokenId < len(reader.unique_tokens):
            next_token = reader.unique_tokens[currentTokenId]
          output_tokens.append(next_token)

  form_sentence(output_tokens)


def form_sentence(output_tokens):
    output_sentence = " "

    if FLAGS.model == "char_model":
        for token in output_tokens:
                output_sentence += token

    elif FLAGS.model ==  "word_model":
        for token in output_tokens:
            if token == "<eos>":
                output_sentence += ("\n")
            else:
                output_sentence += (" " + token)

    print('---- Prediction: \n %s \n----' % (output_sentence))

    with open(out_path,'a') as out:
        out.write(str("\n"+output_sentence))


class WordModel(object):
    init_scale = 0.1
    #learning_rate = 1.0
    learning_rate = 0.003 #0.002
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    sequence_size = 20
    hidden_size = 200
    #hidden_size = 128
    max_epoch = 4
    #max_epoch = 100 #1
    #max_max_epoch = 13
    max_max_epoch = 10000
    keep_prob = 1.0
    #lr_decay = 0.5
    lr_decay = 0.97
    batch_size = 20
    vocab_size = 10000

class WordModelWithDropuout(object):
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    sequence_size = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000

class CharModel(object):
    init_scale = 0.1
    #learning_rate = 1.0
    learning_rate = 0.002
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    sequence_size = 20
    #hidden_size = 200
    hidden_size = 128
    max_epoch = 4
    #max_epoch = 1
    #max_max_epoch = 13
    max_max_epoch = 10000
    keep_prob = 1.0
    #lr_decay = 0.5
    lr_decay = 0.97
    batch_size = 20
    vocab_size = 10000

class CharModelWithDropout(object):
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    sequence_size = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000

if __name__ == "__main__":
  tf.app.run()
