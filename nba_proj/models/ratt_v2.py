# import tensorflow as tf
# import tensorflow.keras as tf_keras
# from official.vision.modeling.layers import nn_blocks
# from official.modeling import activations

# layers = tf_keras.layers


# class RATTHeadV2(tf_keras.Model):
#     def __init__(self, hidden_size=768, num_heads=8, num_layers=2, mlp_dim=128):
#         super().__init__()
#         self.hidden_size = hidden_size
#         SEED = 1234
#         # self.support_proj = layers.Dense(hidden_size, name="support_proj")
#         # self.contrast_proj = layers.Dense(hidden_size, name="contrast_proj")
#         # self.temporal_proj = layers.Dense(hidden_size, name="temporal_proj",activation='relu')

#         self.support_proj = layers.Dense(
#             hidden_size,
#             kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED + 1),
#             bias_initializer="zeros",
#             name="support_proj",
#         )

#         self.contrast_proj = layers.Dense(
#             hidden_size,
#             kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED + 2),
#             bias_initializer="zeros",
#             name="contrast_proj",
#         )

#         self.temporal_proj = layers.Dense(
#             hidden_size,
#             activation='relu',
#             kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED + 3),
#             bias_initializer="zeros",
#             name="temporal_proj",
#         )
#         self.transformer_blocks = []
#         for _ in range(num_layers):
#             block = nn_blocks.TransformerEncoderBlock(
#                 inner_activation=activations.gelu,
#                 num_attention_heads=num_heads,
#                 inner_dim=hidden_size * 4,
#                 output_dropout=0.1,
#                 attention_dropout=0.1,
#                 kernel_regularizer=None,
#                 kernel_initializer="glorot_uniform",
#                 norm_first=True,
#                 stochastic_depth_drop_rate=0.0,
#                 norm_epsilon=1e-6,
#                 layer_scale_init_value=0.0,
#                 transformer_partition_dims=None,
#                 return_attention_scores=True,
#             )
#             self.transformer_blocks.append(block)

#         # learned special tokens
#         self.cls_token = self.add_weight(
#             name="cls_token",
#             shape=(1, 1, hidden_size),
#             initializer=tf.keras.initializers.RandomNormal(stddev=0.02, seed=SEED + 1),
#             trainable=True,
#         )
#         self.support_token = self.add_weight(
#             name="support_token",
#             shape=(1, 1, hidden_size),
#             initializer=tf.keras.initializers.RandomNormal(stddev=0.02, seed=SEED + 2),
#             trainable=True,
#         )
#         self.contrast_token = self.add_weight(
#             name="contrast_token",
#             shape=(1, 1, hidden_size),
#             initializer=tf.keras.initializers.RandomNormal(stddev=0.02, seed=SEED + 3),
#             trainable=True,
#         )
#         self.temporal_token = self.add_weight(
#             name="temporal_token",
#             shape=(1, 1, hidden_size),
#             initializer=tf.keras.initializers.RandomNormal(stddev=0.02, seed=SEED + 4),
#             trainable=True,
#         )
#         # self.cls_token = self.add_weight(
#         #     name="cls_token",
#         #     shape=(1, 1, hidden_size),
#         #     # initializer="random_normal",
#         #     initializer='zeros',
#         #     trainable=True,
#         # )
#         # self.support_token = self.add_weight(
#         #     name="support_token",
#         #     shape=(1, 1, hidden_size),
#         #     # initializer="random_normal",
#         #     initializer='zeros',
#         #     trainable=True,
#         # )
#         # self.contrast_token = self.add_weight(
#         #     name="contrast_token",
#         #     shape=(1, 1, hidden_size),
#         #     # initializer="random_normal",
#         #     initializer='zeros',
#         #     trainable=True,
#         # )
#         # self.temporal_token = self.add_weight(
#         #     name="temporal_token",
#         #     shape=(1, 1, hidden_size),
#         #     # initializer="random_normal",
#         #     initializer='zeros',
#         #     trainable=True,
#         # )

#         # type embeddings
#         self.type_cls = self.add_weight(
#             name="type_cls",
#             shape=(1, 1, hidden_size),
#             # initializer="zeros",
#             initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
#             trainable=True,
#         )
#         self.type_support_summary = self.add_weight(
#             name="type_support_summary",
#             shape=(1, 1, hidden_size),
#             # initializer="zeros",
#             initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
#             trainable=True,
#         )
#         self.type_support = self.add_weight(
#             name="type_support",
#             shape=(1, 1, hidden_size),
#             # initializer="zeros",
#             initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
#             trainable=True,
#         )
#         self.type_contrast_summary = self.add_weight(
#             name="type_contrast_summary",
#             shape=(1, 1, hidden_size),
#             # initializer="zeros",
#             initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
#             trainable=True,
#         )
#         self.type_contrast = self.add_weight(
#             name="type_contrast",
#             shape=(1, 1, hidden_size),
#             # initializer="zeros",
#             initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
#             trainable=True,
#         )
#         self.type_temporal_summary = self.add_weight(
#             name="type_temporal_summary",
#             shape=(1, 1, hidden_size),
#             # initializer="zeros",
#             initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
#             trainable=True,
#         )
#         self.type_temporal = self.add_weight(
#             name="type_temporal",
#             shape=(1, 1, hidden_size),
#             # initializer="zeros",
#             initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
#             trainable=True,
#         )
#         self.type_local = self.add_weight(
#             name="type_local",
#             shape=(1, 1, hidden_size),
#             # initializer="zeros",
#             initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
#             trainable=True,
#         )

#         self.norm = layers.LayerNormalization(epsilon=1e-6)

#         self.classifier = tf_keras.Sequential([
#             layers.Dense(mlp_dim, activation="relu"),
#             layers.Dropout(0.2),
#             layers.Dense(1),
#         ])

#     def call(
#         self,
#         chunk_embs,          # (B, D)
#         support_tokens,      # (B, Ks, D)
#         contrast_tokens,     # (B, Kc, D)
#         temporal_tokens,     # (B, Kt, D)
#         training=False,
#     ):
        

#         B = tf.shape(chunk_embs)[0]
#         Ks = tf.shape(support_tokens)[1]
#         Kc = tf.shape(contrast_tokens)[1]
#         Kt = tf.shape(temporal_tokens)[1]

#         q = tf.expand_dims(chunk_embs, axis=1)              # (B, 1, D)
#         q_tiled = tf.repeat(q, repeats=tf.shape(contrast_tokens)[1], axis=1)   # (B, C, D)

#         support_tokens = self.support_proj(support_tokens)

#         # contrast_tokens = self.contrast_proj(
#         #     tf.concat([contrast_tokens, q_tiled, contrast_tokens - q_tiled], axis=-1)
#         # )

#         # temporal_tokens = self.temporal_proj(
#         #     tf.concat([temporal_tokens, q_tiled, temporal_tokens - q_tiled], axis=-1)
#         # )
#         contrast_tokens = self.contrast_proj(contrast_tokens)
#         temporal_tokens = self.temporal_proj(temporal_tokens)
#         local = tf.expand_dims(chunk_embs, axis=1)  # (B, 1, D)

#         cls = tf.repeat(self.cls_token, repeats=B, axis=0)
#         sup_summary = tf.repeat(self.support_token, repeats=B, axis=0)
#         con_summary = tf.repeat(self.contrast_token, repeats=B, axis=0)
#         tmp_summary = tf.repeat(self.temporal_token, repeats=B, axis=0)

#         x = tf.concat(
#             [
#                 cls,
#                 sup_summary,
#                 support_tokens,
#                 con_summary,
#                 contrast_tokens,
#                 tmp_summary,
#                 temporal_tokens,
#                 local,
#             ],
#             axis=1,
#         )

#         # build matching type embeddings
#         type_parts = [
#             tf.repeat(self.type_cls, repeats=B, axis=0),
#             tf.repeat(self.type_support_summary, repeats=B, axis=0),
#             tf.repeat(self.type_support, repeats=B, axis=0) + tf.zeros((B, Ks, self.hidden_size)),
#             tf.repeat(self.type_contrast_summary, repeats=B, axis=0),
#             tf.repeat(self.type_contrast, repeats=B, axis=0) + tf.zeros((B, Kc, self.hidden_size)),
#             tf.repeat(self.type_temporal_summary, repeats=B, axis=0),
#             tf.repeat(self.type_temporal, repeats=B, axis=0) + tf.zeros((B, Kt, self.hidden_size)),
#             tf.repeat(self.type_local, repeats=B, axis=0),
#         ]
#         type_embs = tf.concat(type_parts, axis=1)

#         x = x + type_embs

#         attn_scores_all = []
#         for block in self.transformer_blocks:
#             x, attn = block(x, training=training)
#             attn_scores_all.append(attn)

#         x = self.norm(x)

#         # fixed positions
#         idx_cls = 0
#         idx_support_summary = 1
#         idx_contrast_summary = 2 + Ks
#         idx_temporal_summary = 3 + Ks + Kc
#         idx_local = 4 + Ks + Kc + Kt

#         cls_out = x[:, idx_cls, :]
#         class_logit = self.classifier(cls_out, training=training)

#         aux = {
#             "support_summary": x[:, idx_support_summary, :],
#             "contrast_summary": x[:, idx_contrast_summary, :],
#             "temporal_summary": x[:, idx_temporal_summary, :],
#             "local_out": x[:, idx_local, :],
#             "attn_scores": attn_scores_all,
#         }

#         # print(aux)
#         last_attn = attn_scores_all[-1]   # shape usually (B, num_heads, T, T)

#         # average over heads
#         attn_mean = tf.reduce_mean(last_attn, axis=1)   # (B, T, T)

#         cls_attn = attn_mean[:, idx_cls, :]             # (B, T)

#         support_attn = tf.reduce_mean(cls_attn[:, 2:2+Ks], axis=1)
#         contrast_attn = tf.reduce_mean(cls_attn[:, 3+Ks:3+Ks+Kc], axis=1)
#         temporal_attn = tf.reduce_mean(cls_attn[:, 4+Ks+Kc:4+Ks+Kc+Kt], axis=1)

#         tf.print("CLS self-attn:", tf.reduce_mean(cls_attn[:, idx_cls]))
#         tf.print("CLS -> support_summary:", tf.reduce_mean(cls_attn[:, idx_support_summary]))
#         tf.print("CLS -> support_tokens:", tf.reduce_mean(support_attn))
#         tf.print("CLS -> contrast_summary:", tf.reduce_mean(cls_attn[:, idx_contrast_summary]))
#         tf.print("CLS -> contrast_tokens:", tf.reduce_mean(contrast_attn))
#         tf.print("CLS -> temporal_summary:", tf.reduce_mean(cls_attn[:, idx_temporal_summary]))
#         tf.print("CLS -> temporal_tokens:", tf.reduce_mean(temporal_attn))
#         tf.print("CLS -> local_out:", tf.reduce_mean(cls_attn[:, idx_local]))
#         return class_logit, cls_out, aux




import tensorflow as tf
import tensorflow.keras as tf_keras
from official.vision.modeling.layers import nn_blocks
from official.modeling import activations
import config_stage2 as config

layers = tf_keras.layers


class RATTHeadV2(tf_keras.Model):
    def __init__(self, hidden_size=768, num_heads=8, num_layers=2, mlp_dim=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        def make_proj(name):
            return tf.keras.Sequential([
                layers.Dense(self.hidden_size * 2, activation='relu'),
                # layers.Dropout(0.2),
                layers.Dense(self.hidden_size),
            ], name=name)
        
        self.query_proj = tf.keras.Sequential([
            layers.Dense(hidden_size),
        ], name="query_proj")
        
        self.support_proj = make_proj("support_proj")
        self.contrast_proj = make_proj("contrast_proj")
        self.temporal_proj = make_proj("temporal_proj")
        # self.query_proj = layers.Dense(hidden_size, name="query_proj")
        # self.support_proj = layers.Dense(hidden_size, name="support_proj")
        # self.contrast_proj = layers.Dense(hidden_size, name="contrast_proj")
        # self.temporal_proj = layers.Dense(hidden_size, name="temporal_proj",activation='relu')


        # self.transformer_blocks = []
        # for _ in range(num_layers):
        #     block = nn_blocks.TransformerEncoderBlock(
        #         inner_activation=activations.gelu,
        #         num_attention_heads=num_heads,
        #         inner_dim=hidden_size * 4,
        #         output_dropout=0.1,
        #         attention_dropout=0.1,
        #         kernel_regularizer=None,
        #         kernel_initializer="glorot_uniform",
        #         norm_first=True,
        #         stochastic_depth_drop_rate=0.0,
        #         norm_epsilon=1e-6,
        #         layer_scale_init_value=0.0,
        #         transformer_partition_dims=None,
        #         return_attention_scores=True,
        #     )
        #     self.transformer_blocks.append(block)

        self.transformer_blocks = []
        for i in range(num_layers):
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
            setattr(self, f"transformer_block_{i}", block)
            # self.transformer_blocks.append(block)

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
            initializer=tf.keras.initializers.RandomNormal(stddev=0.2),
            trainable=True,
        )
        self.type_support_summary = self.add_weight(
            name="type_support_summary",
            shape=(1, 1, hidden_size),
            # initializer="zeros",
            initializer=tf.keras.initializers.RandomNormal(stddev=0.2),
            trainable=True,
        )
        self.type_support = self.add_weight(
            name="type_support",
            shape=(1, 1, hidden_size),
            # initializer="zeros",
            initializer=tf.keras.initializers.RandomNormal(stddev=0.2),
            trainable=True,
        )
        self.type_contrast_summary = self.add_weight(
            name="type_contrast_summary",
            shape=(1, 1, hidden_size),
            # initializer="zeros",
            initializer=tf.keras.initializers.RandomNormal(stddev=0.2),
            trainable=True,
        )
        self.type_contrast = self.add_weight(
            name="type_contrast",
            shape=(1, 1, hidden_size),
            # initializer="zeros",
            initializer=tf.keras.initializers.RandomNormal(stddev=0.2),
            trainable=True,
        )
        self.type_temporal_summary = self.add_weight(
            name="type_temporal_summary",
            shape=(1, 1, hidden_size),
            # initializer="zeros",
            initializer=tf.keras.initializers.RandomNormal(stddev=0.2),
            trainable=True,
        )
        self.type_temporal = self.add_weight(
            name="type_temporal",
            shape=(1, 1, hidden_size),
            # initializer="zeros",
            initializer=tf.keras.initializers.RandomNormal(stddev=0.2),
            trainable=True,
        )
        self.type_local = self.add_weight(
            name="type_local",
            shape=(1, 1, hidden_size),
            # initializer="zeros",
            initializer=tf.keras.initializers.RandomNormal(stddev=0.2),
            trainable=True,
        )
        self.support_pos = self.add_weight(
            name="support_pos",
            shape=(1, config.K_SIM, self.hidden_size),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.2),
            trainable=True,
        )
        self.contrast_pos = self.add_weight(
            name="contrast_pos",
            shape=(1, config.K_CONTRAST, self.hidden_size),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.2),
            trainable=True,
        )
        self.temporal_pos = self.add_weight(
            name="temporal_pos",
            shape=(1, config.K_TEMPORAL, self.hidden_size),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.2),
            trainable=True,
        )

        self.norm = layers.LayerNormalization(epsilon=1e-6)

        self.classifier = tf_keras.Sequential([
            layers.Dense(mlp_dim*2, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(3),
        ])

    def debug_forward(
        self,
        chunk_embs,
        support_tokens,
        contrast_tokens,
        temporal_tokens,
        training=False,
    ):
        B = tf.shape(chunk_embs)[0]
        Ks = tf.shape(support_tokens)[1]
        Kc = tf.shape(contrast_tokens)[1]
        Kt = tf.shape(temporal_tokens)[1]

        q_raw = tf.expand_dims(chunk_embs, axis=1)
        q_proj = self.query_proj(q_raw)
        local = q_raw + q_proj

        support_proj = self.support_proj(support_tokens)
        contrast_proj = self.contrast_proj(contrast_tokens)
        temporal_proj = self.temporal_proj(temporal_tokens)

        # support_tokens = support_tokens + self.support_pos
        # contrast_tokens = contrast_tokens + self.contrast_pos
        # temporal_tokens = temporal_tokens + self.temporal_pos

        support_tokens = support_tokens + self.support_pos[:, :Ks, :]
        contrast_tokens = contrast_tokens + self.contrast_pos[:, :Kc, :]
        temporal_tokens = temporal_tokens + self.temporal_pos[:, :Kt, :]

        cls = tf.repeat(self.cls_token, repeats=B, axis=0)
        sup_summary = tf.repeat(self.support_token, repeats=B, axis=0)
        con_summary = tf.repeat(self.contrast_token, repeats=B, axis=0)
        tmp_summary = tf.repeat(self.temporal_token, repeats=B, axis=0)

        x = tf.concat(
            [
                cls,
                sup_summary,
                support_proj,
                con_summary,
                contrast_proj,
                tmp_summary,
                temporal_proj,
                local,
            ],
            axis=1,
        )

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
        x_plus_type = x + type_embs

        block_outs = []
        attn_outs = []
        x_block = x_plus_type
        for block in self.transformer_blocks:
            x_block, attn = block(x_block, training=training)
            block_outs.append(x_block)
            attn_outs.append(attn)

        x_norm = self.norm(x_block)

        idx_cls = 0
        idx_local = 4 + Ks + Kc + Kt

        cls_out = x_norm[:, idx_cls, :]
        local_out = x_norm[:, idx_local, :]
        fused_out = tf.concat([cls_out, 5 * local_out], axis=-1)
        logits = self.classifier(fused_out, training=training)

        return {
            "q_proj": q_proj,
            "local": local,
            "support_proj": support_proj,
            "contrast_proj": contrast_proj,
            "temporal_proj": temporal_proj,
            "x_plus_type": x_plus_type,
            "block_outs": block_outs,
            "attn_outs": attn_outs,
            "x_norm": x_norm,
            "fused_out": fused_out,
            "logits": logits,
        }

    def call(
        self,
        chunk_embs,          # (B, D)
        support_tokens,      # (B, Ks, D)
        contrast_tokens,     # (B, Kc, D)
        temporal_tokens,     # (B, Kt, D)
        support_mask=None,   # (B, Ks) bool
        contrast_mask=None,  # (B, Kc) bool
        temporal_mask=None,  # (B, Kt) bool
        training=False,
    ):
        B = tf.shape(chunk_embs)[0]
        Ks = tf.shape(support_tokens)[1]
        Kc = tf.shape(contrast_tokens)[1]
        Kt = tf.shape(temporal_tokens)[1]

        q_raw = tf.expand_dims(chunk_embs, axis=1)              # (B, 1, D)
        q_tiled = tf.repeat(q_raw, repeats=tf.shape(contrast_tokens)[1], axis=1)

        q_proj = self.query_proj(q_raw)
        local = q_raw + q_proj

        support_tokens = self.support_proj(support_tokens)
        contrast_tokens = self.contrast_proj(contrast_tokens)
        temporal_tokens = self.temporal_proj(temporal_tokens)

        support_tokens = support_tokens + self.support_pos[:, :Ks, :]
        contrast_tokens = contrast_tokens + self.contrast_pos[:, :Kc, :]
        temporal_tokens = temporal_tokens + self.temporal_pos[:, :Kt, :]

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

        # build sequence mask — True = real token, False = pad
        # cls, summaries, and local are always real so they get ones
        ones_1 = tf.ones((B, 1), dtype=tf.bool)

        if support_mask is None:
            support_mask = tf.ones((B, Ks), dtype=tf.bool)
        if contrast_mask is None:
            contrast_mask = tf.ones((B, Kc), dtype=tf.bool)
        if temporal_mask is None:
            temporal_mask = tf.ones((B, Kt), dtype=tf.bool)

        seq_mask = tf.concat([
            ones_1,          # cls
            ones_1,          # sup_summary
            support_mask,    # support tokens  (B, Ks)
            ones_1,          # con_summary
            contrast_mask,   # contrast tokens (B, Kc)
            ones_1,          # tmp_summary
            temporal_mask,   # temporal tokens (B, Kt)
            ones_1,          # local
        ], axis=1)  # (B, seq_len)

        attn_scores_all = []
        for i in range(self.num_layers):
            block = getattr(self, f"transformer_block_{i}")
            # expand seq_mask from (B, seq_len) to (B, 1, 1, seq_len) for broadcasting
            # across heads and query positions
            mask_4d = seq_mask[:, tf.newaxis, tf.newaxis, :]  # (B, 1, 1, seq_len)
            x, attn = block([x, mask_4d], training=training)
            attn_scores_all.append(attn)
        x = self.norm(x)

        # fixed positions
        idx_cls = 0
        idx_support_summary = 1
        idx_contrast_summary = 2 + Ks
        idx_temporal_summary = 3 + Ks + Kc
        idx_local = 4 + Ks + Kc + Kt

        cls_out = x[:, idx_cls, :]
        local_out = x[:, idx_local, :]

        alpha = 0
        fused_out = tf.concat([cls_out, alpha * local_out], axis=-1)
        class_logit = self.classifier(fused_out, training=training)

        aux = {
            "support_summary": x[:, idx_support_summary, :],
            "contrast_summary": x[:, idx_contrast_summary, :],
            "temporal_summary": x[:, idx_temporal_summary, :],
            "local_out": x[:, idx_local, :],
            "attn_scores": attn_scores_all,
        }

        last_attn = attn_scores_all[-1]   # (B, num_heads, T, T)
        attn_mean = tf.reduce_mean(last_attn, axis=1)   # (B, T, T)
        cls_attn = attn_mean[:, idx_cls, :]             # (B, T)

        cls_self = cls_attn[:, idx_cls]
        cls_to_support_summary = cls_attn[:, idx_support_summary]
        cls_to_support = tf.reduce_mean(cls_attn[:, 2:2+Ks], axis=1)
        cls_to_contrast_summary = cls_attn[:, idx_contrast_summary]
        cls_to_contrast = tf.reduce_mean(cls_attn[:, 3+Ks:3+Ks+Kc], axis=1)
        cls_to_temporal_summary = cls_attn[:, idx_temporal_summary]
        cls_to_temporal = tf.reduce_mean(cls_attn[:, 4+Ks+Kc:4+Ks+Kc+Kt], axis=1)
        cls_to_local = cls_attn[:, idx_local]
        
        tf.print("seq_mask real token count:", tf.reduce_sum(tf.cast(seq_mask, tf.int32), axis=1))
        tf.print(
            "cls_attn | "
            "self:", tf.reduce_mean(cls_self),
            "sup_sum:", tf.reduce_mean(cls_to_support_summary),
            "sup:", tf.reduce_mean(cls_to_support),
            "con_sum:", tf.reduce_mean(cls_to_contrast_summary),
            "con:", tf.reduce_mean(cls_to_contrast),
            "tmp_sum:", tf.reduce_mean(cls_to_temporal_summary),
            "tmp:", tf.reduce_mean(cls_to_temporal),
            "local:", tf.reduce_mean(cls_to_local),
            summarize=-1,
        )

        return class_logit, cls_out, aux

    # def call(
    #     self,
    #     chunk_embs,          # (B, D)
    #     support_tokens,      # (B, Ks, D)
    #     contrast_tokens,     # (B, Kc, D)
    #     temporal_tokens,     # (B, Kt, D)
    #     training=False,
    # ):
        

    #     B = tf.shape(chunk_embs)[0]
    #     Ks = tf.shape(support_tokens)[1]
    #     Kc = tf.shape(contrast_tokens)[1]
    #     Kt = tf.shape(temporal_tokens)[1]

    #     q_raw = tf.expand_dims(chunk_embs, axis=1)              # (B, 1, D)
    #     q_tiled = tf.repeat(q_raw, repeats=tf.shape(contrast_tokens)[1], axis=1)   # (B, C, D)

    #     q_proj = self.query_proj(q_raw)    
    #     local = q_raw + q_proj

    #     support_tokens = self.support_proj(support_tokens)

    #     # contrast_tokens = self.contrast_proj(
    #     #     tf.concat([contrast_tokens, q_tiled, contrast_tokens - q_tiled], axis=-1)
    #     # )

    #     # temporal_tokens = self.temporal_proj(
    #     #     tf.concat([temporal_tokens, q_tiled, temporal_tokens - q_tiled], axis=-1)
    #     # )

    #     contrast_tokens = self.contrast_proj(contrast_tokens)
    #     temporal_tokens = self.temporal_proj(temporal_tokens)


    #     # q_sup = tf.repeat(q_raw, repeats=Ks, axis=1)
    #     # q_con = tf.repeat(q_raw, repeats=Kc, axis=1)
    #     # q_tmp = tf.repeat(q_raw, repeats=Kt, axis=1)

    #     # support_tokens = self.support_proj(
    #     #     tf.concat([support_tokens, q_sup, support_tokens - q_sup], axis=-1)
    #     # )
    #     # contrast_tokens = self.contrast_proj(
    #     #     tf.concat([contrast_tokens, q_con, contrast_tokens - q_con], axis=-1)
    #     # )
    #     # temporal_tokens = self.temporal_proj(
    #     #     tf.concat([temporal_tokens, q_tmp, temporal_tokens - q_tmp], axis=-1)
    #     # )
    #     # local = tf.expand_dims(chunk_embs, axis=1)  # (B, 1, D)

    #     cls = tf.repeat(self.cls_token, repeats=B, axis=0)
    #     sup_summary = tf.repeat(self.support_token, repeats=B, axis=0)
    #     con_summary = tf.repeat(self.contrast_token, repeats=B, axis=0)
    #     tmp_summary = tf.repeat(self.temporal_token, repeats=B, axis=0)

    #     # x = tf.concat(
    #     #     [
    #     #         cls,
    #     #         sup_summary,
    #     #         support_tokens,
    #     #         con_summary,
    #     #         contrast_tokens,
    #     #         tmp_summary,
    #     #         temporal_tokens,
    #     #         local,
    #     #     ],
    #     #     axis=1,
    #     # )
    #     x = tf.concat(
    #         [
    #             cls,
    #             sup_summary,
    #             support_tokens,
    #             con_summary,
    #             contrast_tokens,
    #             tmp_summary,
    #             temporal_tokens,
    #             local,
    #         ],
    #         axis=1,
    #     )


    #     # build matching type embeddings
    #     type_parts = [
    #         tf.repeat(self.type_cls, repeats=B, axis=0),
    #         tf.repeat(self.type_support_summary, repeats=B, axis=0),
    #         tf.repeat(self.type_support, repeats=B, axis=0) + tf.zeros((B, Ks, self.hidden_size)),
    #         tf.repeat(self.type_contrast_summary, repeats=B, axis=0),
    #         tf.repeat(self.type_contrast, repeats=B, axis=0) + tf.zeros((B, Kc, self.hidden_size)),
    #         tf.repeat(self.type_temporal_summary, repeats=B, axis=0),
    #         tf.repeat(self.type_temporal, repeats=B, axis=0) + tf.zeros((B, Kt, self.hidden_size)),
    #         tf.repeat(self.type_local, repeats=B, axis=0),
    #     ]
    #     type_embs = tf.concat(type_parts, axis=1)

    #     x = x + type_embs

    #     # attn_scores_all = []

    #     # for block in self.transformer_blocks:
    #     #     x, attn = block(x, training=training)
    #     #     attn_scores_all.append(attn)

    #     attn_scores_all = []
    #     for i in range(self.num_layers):
    #         block = getattr(self, f"transformer_block_{i}")
    #         x, attn = block(x, training=training)
    #         attn_scores_all.append(attn)
    #     x = self.norm(x)

    #     # fixed positions
    #     idx_cls = 0
    #     idx_support_summary = 1
    #     idx_contrast_summary = 2 + Ks
    #     idx_temporal_summary = 3 + Ks + Kc
    #     idx_local = 4 + Ks + Kc + Kt

        

    #     cls_out = x[:, idx_cls, :]
    #     local_out = x[:, idx_local, :]

    #     # fused_out = tf.concat([cls_out, local_out],axis=-1)
    #     alpha = 5
    #     fused_out = tf.concat([cls_out, alpha * local_out], axis=-1)
    #     class_logit = self.classifier(fused_out, training=training)

    #     aux = {
    #         "support_summary": x[:, idx_support_summary, :],
    #         "contrast_summary": x[:, idx_contrast_summary, :],
    #         "temporal_summary": x[:, idx_temporal_summary, :],
    #         "local_out": x[:, idx_local, :],
    #         "attn_scores": attn_scores_all,
    #     }

    #     # print(aux)
    #     last_attn = attn_scores_all[-1]   # shape usually (B, num_heads, T, T)

    #     # average over heads
    #     attn_mean = tf.reduce_mean(last_attn, axis=1)   # (B, T, T)

    #     cls_attn = attn_mean[:, idx_cls, :]             # (B, T)
        
    #     support_attn = tf.reduce_mean(cls_attn[:, 2:2+Ks], axis=1)
    #     contrast_attn = tf.reduce_mean(cls_attn[:, 3+Ks:3+Ks+Kc], axis=1)
    #     temporal_attn = tf.reduce_mean(cls_attn[:, 4+Ks+Kc:4+Ks+Kc+Kt], axis=1)

    #     cls_self = cls_attn[:, idx_cls]
    #     cls_to_support_summary = cls_attn[:, idx_support_summary]
    #     cls_to_support = tf.reduce_mean(cls_attn[:, 2:2+Ks], axis=1)
    #     cls_to_contrast_summary = cls_attn[:, idx_contrast_summary]
    #     cls_to_contrast = tf.reduce_mean(cls_attn[:, 3+Ks:3+Ks+Kc], axis=1)
    #     cls_to_temporal_summary = cls_attn[:, idx_temporal_summary]
    #     cls_to_temporal = tf.reduce_mean(cls_attn[:, 4+Ks+Kc:4+Ks+Kc+Kt], axis=1)
    #     cls_to_local = cls_attn[:, idx_local]

    #     # batch-mean each
    #     attn_dict = {
    #         "self":     tf.reduce_mean(cls_self),
    #         "sup_sum":  tf.reduce_mean(cls_to_support_summary),
    #         "support":  tf.reduce_mean(cls_to_support),
    #         "con_sum":  tf.reduce_mean(cls_to_contrast_summary),
    #         "contrast": tf.reduce_mean(cls_to_contrast),
    #         "tmp_sum":  tf.reduce_mean(cls_to_temporal_summary),
    #         "temporal": tf.reduce_mean(cls_to_temporal),
    #         "local":    tf.reduce_mean(cls_to_local),
    #     }

    #     tf.print(
    #         "cls_attn | "
    #         "self:", attn_dict["self"],
    #         "sup_sum:", attn_dict["sup_sum"],
    #         "sup:", attn_dict["support"],
    #         "con_sum:", attn_dict["con_sum"],
    #         "con:", attn_dict["contrast"],
    #         "tmp_sum:", attn_dict["tmp_sum"],
    #         "tmp:", attn_dict["temporal"],
    #         "local:", attn_dict["local"],
    #         summarize=-1,
    #     )
    #     return class_logit, cls_out, aux