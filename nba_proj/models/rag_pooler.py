import tensorflow as tf
import tensorflow.keras as tf_keras

layers = tf_keras.layers

class RetrievalMultiQueryPooler(layers.Layer): 
  def __init__(self,hidden_size,num_queries):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_queries = num_queries

    # add weight just adds this learnable set of parameters 
    # to the execution graph (i think)
    self.query = self.add_weight(
      shape = (num_queries, hidden_size),
      initializer = 'glorot_uniform',
      trainable = True, 
      name = 'retrieval_queries'
    )

  def call(self, retrieved):

    # retrieved is a tensor of shape (B, R, D)
    B = tf.shape(retrieved)[0]

    # tf.tile duplicates things in the first parameter, as many times
    # as specified in the second parameter 
    # so if the first param was [[1,2,3],[4,5,6]] and the second was [1,3]
    # the result is [[1,2,3,1,2,3,1,2,3],[4,5,6,4,5,6,4,5,6]] 
    # replicated once a row, 3 times across each row

    # tf.expand_dims expands the dimension (just adds another dimension)
    queries = tf.tile(tf.expand_dims(self.query, 0), [B, 1, 1])

    # attention scores (q * k transpose)
    scores = tf.matmul(queries, retrieved, transpose_b = True)

    # softmax the attention scores (q * ktranspose) / sqrt(n)
    weights = tf.nn.softmax(scores, axis = -1)

    retrieval_tokens = tf.matmul(weights, retrieved)
    return retrieval_tokens