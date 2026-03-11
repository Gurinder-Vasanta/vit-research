import tensorflow as tf
import tensorflow.keras as tf_keras
from official.vision.modeling.layers import nn_blocks
from official.modeling import activations
from models.rag_pooler import RetrievalMultiQueryPooler

layers = tf_keras.layers

class RATTHead(tf_keras.Model):
  def __init__(self, hidden_size=768, num_queries=4, num_layers=2, num_heads=4):
    super().__init__()
    self.pooler = RetrievalMultiQueryPooler(hidden_size, num_queries)
    self.transformer_blocks = []
    self.hidden_size = hidden_size

    for i in range(num_layers):
        block = nn_blocks.TransformerEncoderBlock(
            inner_activation=activations.gelu,
            num_attention_heads=num_heads,
            inner_dim=hidden_size * 4,
            output_dropout=0.1,
            attention_dropout=0.1,
            kernel_regularizer=None,
            kernel_initializer='glorot_uniform',
            norm_first=True,
            stochastic_depth_drop_rate=0.0,
            norm_epsilon=1e-6,
            layer_scale_init_value=0.0,
            transformer_partition_dims=None,
            return_attention_scores=True,
        )
        self.transformer_blocks.append(block)

    self.norm = layers.LayerNormalization(epsilon=1e-6)

    self.classifier = tf.keras.Sequential([
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])

    self.cls_type = self.add_weight(
        shape=(1, 1, hidden_size),
        initializer="zeros",
        trainable=True
    )

    self.ret_type = self.add_weight(
        shape=(1, 1, hidden_size),
        initializer="zeros",
        trainable=True
    )

    # self.pos_embedding = self.add_weight(
    #     shape=(1, 1 + num_queries, hidden_size),
    #     initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
    #     trainable=True
    # )

    self.max_tokens = 128   # or 256 or whatever upper bound you want

    self.pos_embedding = self.add_weight(
        shape=(1, self.max_tokens, hidden_size),
        initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
        trainable=True
    )
    # self.pos_embedding = None

    # self.classifier = layers.Dense(1)  # for binary make/miss


# self, cls_embeddings, retrieved_embeddings, disable_cls=True, training=False, use_retrieval=True
    # def call(self, retrieved_embeddings, training=False):
#   def call(self, cls_embeddings, retrieved_embeddings, disable_cls=True, training=False, use_retrieval=True):
  
#       # tokens = retrieved only
#       x = retrieved_embeddings + self.ret_type
  
#       seq_len = tf.shape(x)[1]
#       pos_emb = tf.zeros((1, seq_len, self.hidden_size))
#       x = x + pos_emb
  
#       attention_scores_all = []
#       for block in self.transformer_blocks:
#           x, attn_scores = block(x, training=training)
#           attention_scores_all.append(attn_scores)
  
#       x = self.norm(x)
  
#       pooled = tf.reduce_mean(x, axis=1)
#       logits = self.classifier(pooled)
  
#       junk = []
#       return logits, junk, attention_scores_all

  
  
  def call(self, cls_embeddings, retrieved_embeddings, disable_cls=True, training=False, use_retrieval=True):

    # x = retrieved_embeddings + self.ret_type   # tokens only
    cls_token = tf.expand_dims(cls_embeddings, axis=1) + self.cls_type
    ret_tokens = retrieved_embeddings + self.ret_type

    x = tf.concat([cls_token, ret_tokens], axis=1)

    x = x + self.pos_embedding[:, :tf.shape(x)[1], :]

    attention_scores_all = []
    for block in self.transformer_blocks:
        x, attn_scores = block(x, training=training)
        attention_scores_all.append(attn_scores)

    x = self.norm(x)

    attn = attention_scores_all[-1]

    # importance = tf.reduce_mean(attn, axis=[1,2])
    # importance = tf.nn.softmax(importance, axis=-1)

    # fused = tf.reduce_sum(importance[:, :, None] * x, axis=1)

    fused = x[:, 0, :]

    # fused = tf.reduce_mean(x, axis=1)   # pool

    logits = self.classifier(fused,training=training)

    return logits, fused, attention_scores_all
  
#   def call(self, cls_embeddings, retrieved_embeddings, disable_cls=True, training=False, use_retrieval=True):
#     """
#     cls_embeddings: (B, D)
#     retrieved_embeddings: (B, k, D)
#     """

#     # print("CLS embeddings:", 
#     #   float(tf.reduce_mean(cls_embeddings)), 
#     #   float(tf.math.reduce_std(cls_embeddings)))

#     # print("RET embeddings:", 
#     #   float(tf.reduce_mean(retrieved_embeddings)), 
#     #   float(tf.math.reduce_std(retrieved_embeddings)))
#     # 73 57 52 667
#     # input('stop')

#     # cls_embeddings = tf.zeros_like(cls_embeddings)
#     # retrieved_embeddings = tf.zeros_like(retrieved_embeddings)

#     # (B, 1, D)
#     cls_tokens = tf.expand_dims(cls_embeddings, axis=1)

#     cls_tokens = cls_tokens + self.cls_type

#     if use_retrieval: 
#         # (B, R, D)
#         # retrieval_tokens = self.pooler(retrieved_embeddings)
#         retrieval_tokens = retrieved_embeddings
#         retrieval_tokens = retrieval_tokens + self.ret_type
#         # concat → (B, 1+R, D)
#         x = tf.concat([cls_tokens, retrieval_tokens], axis=1)

#         x = x + self.pos_embedding
#     else: 
#        x = cls_tokens + self.pos_embedding[:, :1, :]
#         # x = cls_tokens
    
#     # seq_len = tf.shape(x)[1]
#     # pos_emb = tf.zeros((1, seq_len, self.hidden_size))
#     # x = x + pos_emb

#     attention_scores_all = []
#     for block in self.transformer_blocks:
#         x, attn_scores = block(x, training=training)
#         attention_scores_all.append(attn_scores)

#         # cls_to_ret = attn_scores[:, :, 0, 1:]   # (B, heads, k)
#         # importance = tf.reduce_mean(cls_to_ret, axis=1)    # (B, k)
#         # tf.print("importance:", importance)
#         # print(importance)

#     # input(attention_scores_all)
#     x = self.norm(x)

#     # if disable_cls:
#     #     fused_cls = tf.reduce_mean(x[:, 1:], axis=1)
#     # else:
#     #     fused_cls = x[:, 0]
#     fused_cls = x[:, 0]        # (B, D)
#     logits = self.classifier(fused_cls)  # (B, 1)

#     return logits, fused_cls, attention_scores_all