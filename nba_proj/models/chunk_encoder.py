import tensorflow as tf
import tensorflow.keras as tf_keras
from official.vision.modeling.layers import nn_blocks
from official.modeling import activations

layers = tf_keras.layers


class ChunkEncoder(tf_keras.Model):
    def __init__(self, hidden_size=768, num_layers=3, num_heads=8, max_frames=24):
        super().__init__()
        self.transformer_blocks = []
        self.hidden_size = hidden_size
        self.max_tokens = 1 + max_frames   # 1 CLS + T frame tokens

        for _ in range(num_layers):
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

        self.class_head = tf.keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1)  # chunk proxy logit
        ])

        self.cls_token = self.add_weight(
            shape=(1, 1, hidden_size),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
            name="cls_token"
        )

        self.pos_embedding = self.add_weight(
            shape=(1, self.max_tokens, hidden_size),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
            name="pos_embedding"
        )

    def call(self, frame_embeddings, training=False, return_attention=False):
        """
        frame_embeddings: (B, T, 768)
        returns:
            chunk_emb:   (B, 768)
            class_logit: (B, 1)
            optionally attention_scores_all
        """
        B = tf.shape(frame_embeddings)[0]
        T = tf.shape(frame_embeddings)[1]

        if frame_embeddings.shape.rank != 3:
            raise ValueError(
                f"Expected frame_embeddings to have shape (B, T, D), got {frame_embeddings.shape}"
            )

        # learned CLS token repeated across the batch
        cls_token = tf.repeat(self.cls_token, repeats=B, axis=0)  # (B, 1, D)

        # sequence = [CLS, frame_1, ..., frame_T]
        x = tf.concat([cls_token, frame_embeddings], axis=1)      # (B, 1+T, D)

        # add positional embeddings
        x = x + self.pos_embedding[:, :T + 1, :]

        attention_scores_all = []
        for block in self.transformer_blocks:
            x, attn_scores = block(x, training=training)
            attention_scores_all.append(attn_scores)

        x = self.norm(x)

        # CLS output is the chunk embedding
        chunk_emb = x[:, 0, :]                                    # (B, D)
        chunk_emb = tf.nn.l2_normalize(chunk_emb, axis=-1)

        class_logit = self.class_head(chunk_emb, training=training)

        if return_attention:
            return chunk_emb, class_logit, attention_scores_all

        return chunk_emb, class_logit
