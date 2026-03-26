import tensorflow as tf
import tensorflow.keras as tf_keras
from official.vision.modeling.layers import nn_blocks
from official.modeling import activations

layers = tf_keras.layers


class RATTHeadV2(tf_keras.Model):
    def __init__(self, hidden_size=768, num_heads=8, num_layers=2, mlp_dim=128):
        super().__init__()
        self.hidden_size = hidden_size

        self.support_proj = layers.Dense(hidden_size, name="support_proj",activation='relu')
        self.contrast_proj = layers.Dense(hidden_size, name="contrast_proj")
        self.temporal_proj = layers.Dense(hidden_size, name="temporal_proj")

        self.transformer_blocks = []
        for _ in range(num_layers):
            block = nn_blocks.TransformerEncoderBlock(
                inner_activation=activations.gelu,
                num_attention_heads=num_heads,
                inner_dim=hidden_size * 4,
                output_dropout=0.1,
                attention_dropout=0.1,
                kernel_regularizer=None,
                kernel_initializer="glorot_uniform",
                norm_first=True,
                stochastic_depth_drop_rate=0.0,
                norm_epsilon=1e-6,
                layer_scale_init_value=0.0,
                transformer_partition_dims=None,
                return_attention_scores=True,
            )
            self.transformer_blocks.append(block)

        # learned special tokens
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, hidden_size),
            initializer="random_normal",
            trainable=True,
        )
        self.support_token = self.add_weight(
            name="support_token",
            shape=(1, 1, hidden_size),
            initializer="random_normal",
            trainable=True,
        )
        self.contrast_token = self.add_weight(
            name="contrast_token",
            shape=(1, 1, hidden_size),
            initializer="random_normal",
            trainable=True,
        )
        self.temporal_token = self.add_weight(
            name="temporal_token",
            shape=(1, 1, hidden_size),
            initializer="random_normal",
            trainable=True,
        )

        # type embeddings
        self.type_cls = self.add_weight(
            name="type_cls",
            shape=(1, 1, hidden_size),
            # initializer="zeros",
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        self.type_support_summary = self.add_weight(
            name="type_support_summary",
            shape=(1, 1, hidden_size),
            # initializer="zeros",
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        self.type_support = self.add_weight(
            name="type_support",
            shape=(1, 1, hidden_size),
            # initializer="zeros",
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        self.type_contrast_summary = self.add_weight(
            name="type_contrast_summary",
            shape=(1, 1, hidden_size),
            # initializer="zeros",
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        self.type_contrast = self.add_weight(
            name="type_contrast",
            shape=(1, 1, hidden_size),
            # initializer="zeros",
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        self.type_temporal_summary = self.add_weight(
            name="type_temporal_summary",
            shape=(1, 1, hidden_size),
            # initializer="zeros",
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        self.type_temporal = self.add_weight(
            name="type_temporal",
            shape=(1, 1, hidden_size),
            # initializer="zeros",
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        self.type_local = self.add_weight(
            name="type_local",
            shape=(1, 1, hidden_size),
            # initializer="zeros",
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )

        self.norm = layers.LayerNormalization(epsilon=1e-6)

        self.classifier = tf_keras.Sequential([
            layers.Dense(mlp_dim, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1),
        ])

    def call(
        self,
        chunk_embs,          # (B, D)
        support_tokens,      # (B, Ks, D)
        contrast_tokens,     # (B, Kc, D)
        temporal_tokens,     # (B, Kt, D)
        training=False,
    ):
        

        B = tf.shape(chunk_embs)[0]
        Ks = tf.shape(support_tokens)[1]
        Kc = tf.shape(contrast_tokens)[1]
        Kt = tf.shape(temporal_tokens)[1]

        q = tf.expand_dims(chunk_embs, axis=1)              # (B, 1, D)
        q_tiled = tf.repeat(q, repeats=tf.shape(contrast_tokens)[1], axis=1)   # (B, C, D)

        support_tokens = self.support_proj(support_tokens)

        # contrast_tokens = self.contrast_proj(
        #     tf.concat([contrast_tokens, q_tiled, contrast_tokens - q_tiled], axis=-1)
        # )

        # temporal_tokens = self.temporal_proj(
        #     tf.concat([temporal_tokens, q_tiled, temporal_tokens - q_tiled], axis=-1)
        # )
        contrast_tokens = self.contrast_proj(contrast_tokens)
        temporal_tokens = self.temporal_proj(temporal_tokens)
        local = tf.expand_dims(chunk_embs, axis=1)  # (B, 1, D)

        cls = tf.repeat(self.cls_token, repeats=B, axis=0)
        sup_summary = tf.repeat(self.support_token, repeats=B, axis=0)
        con_summary = tf.repeat(self.contrast_token, repeats=B, axis=0)
        tmp_summary = tf.repeat(self.temporal_token, repeats=B, axis=0)

        x = tf.concat(
            [
                cls,
                sup_summary,
                support_tokens,
                con_summary,
                contrast_tokens,
                tmp_summary,
                temporal_tokens,
                local,
            ],
            axis=1,
        )

        # build matching type embeddings
        type_parts = [
            tf.repeat(self.type_cls, repeats=B, axis=0),
            tf.repeat(self.type_support_summary, repeats=B, axis=0),
            tf.repeat(self.type_support, repeats=B, axis=0) + tf.zeros((B, Ks, self.hidden_size)),
            tf.repeat(self.type_contrast_summary, repeats=B, axis=0),
            tf.repeat(self.type_contrast, repeats=B, axis=0) + tf.zeros((B, Kc, self.hidden_size)),
            tf.repeat(self.type_temporal_summary, repeats=B, axis=0),
            tf.repeat(self.type_temporal, repeats=B, axis=0) + tf.zeros((B, Kt, self.hidden_size)),
            tf.repeat(self.type_local, repeats=B, axis=0),
        ]
        type_embs = tf.concat(type_parts, axis=1)

        x = x + type_embs

        attn_scores_all = []
        for block in self.transformer_blocks:
            x, attn = block(x, training=training)
            attn_scores_all.append(attn)

        x = self.norm(x)

        # fixed positions
        idx_cls = 0
        idx_support_summary = 1
        idx_contrast_summary = 2 + Ks
        idx_temporal_summary = 3 + Ks + Kc
        idx_local = 4 + Ks + Kc + Kt

        cls_out = x[:, idx_cls, :]
        class_logit = self.classifier(cls_out, training=training)

        aux = {
            "support_summary": x[:, idx_support_summary, :],
            "contrast_summary": x[:, idx_contrast_summary, :],
            "temporal_summary": x[:, idx_temporal_summary, :],
            "local_out": x[:, idx_local, :],
            "attn_scores": attn_scores_all,
        }

        # print(aux)
        last_attn = attn_scores_all[-1]   # shape usually (B, num_heads, T, T)

        # average over heads
        attn_mean = tf.reduce_mean(last_attn, axis=1)   # (B, T, T)

        cls_attn = attn_mean[:, idx_cls, :]             # (B, T)

        support_attn = tf.reduce_mean(cls_attn[:, 2:2+Ks], axis=1)
        contrast_attn = tf.reduce_mean(cls_attn[:, 3+Ks:3+Ks+Kc], axis=1)
        temporal_attn = tf.reduce_mean(cls_attn[:, 4+Ks+Kc:4+Ks+Kc+Kt], axis=1)

        tf.print("CLS self-attn:", tf.reduce_mean(cls_attn[:, idx_cls]))
        tf.print("CLS -> support_summary:", tf.reduce_mean(cls_attn[:, idx_support_summary]))
        tf.print("CLS -> support_tokens:", tf.reduce_mean(support_attn))
        tf.print("CLS -> contrast_summary:", tf.reduce_mean(cls_attn[:, idx_contrast_summary]))
        tf.print("CLS -> contrast_tokens:", tf.reduce_mean(contrast_attn))
        tf.print("CLS -> temporal_summary:", tf.reduce_mean(cls_attn[:, idx_temporal_summary]))
        tf.print("CLS -> temporal_tokens:", tf.reduce_mean(temporal_attn))
        tf.print("CLS -> local_out:", tf.reduce_mean(cls_attn[:, idx_local]))
        return class_logit, cls_out, aux