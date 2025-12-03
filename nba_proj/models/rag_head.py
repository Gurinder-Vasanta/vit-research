import tensorflow as tf
import tensorflow.keras as tf_keras
from official.vision.modeling.layers import nn_blocks
from official.modeling import activations
from models.rag_pooler import RetrievalMultiQueryPooler

layers = tf_keras.layers

class RAGHead(tf_keras.Model):
  def __init__(self, hidden_size=768, num_queries=4, num_layers=2, num_heads=4):
    super().__init__()
    self.pooler = RetrievalMultiQueryPooler(hidden_size, num_queries)
    self.transformer_blocks = []

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
            return_attention_scores=False,
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

    self.pos_embedding = self.add_weight(
        shape=(1, 1 + num_queries, hidden_size),
        initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
        trainable=True
    )

    # self.classifier = layers.Dense(1)  # for binary make/miss

  def call(self, cls_embeddings, retrieved_embeddings, training=False):
    """
    cls_embeddings: (B, D)
    retrieved_embeddings: (B, k, D)
    """

    # print("CLS embeddings:", 
    #   float(tf.reduce_mean(cls_embeddings)), 
    #   float(tf.math.reduce_std(cls_embeddings)))

    # print("RET embeddings:", 
    #   float(tf.reduce_mean(retrieved_embeddings)), 
    #   float(tf.math.reduce_std(retrieved_embeddings)))
    
    # input('stop')
    # (B, R, D)
    retrieval_tokens = self.pooler(retrieved_embeddings)

    # (B, 1, D)
    cls_tokens = tf.expand_dims(cls_embeddings, axis=1)

    cls_tokens = cls_tokens + self.cls_type
    retrieval_tokens = retrieval_tokens + self.ret_type

    # concat → (B, 1+R, D)
    x = tf.concat([cls_tokens, retrieval_tokens], axis=1)

    x = x + self.pos_embedding
    for block in self.transformer_blocks:
        x = block(x, training=training)

    x = self.norm(x)

    fused_cls = x[:, 0]        # (B, D)
    logits = self.classifier(fused_cls)  # (B, 1)

    return logits, fused_cls