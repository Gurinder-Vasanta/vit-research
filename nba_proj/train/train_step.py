import tensorflow as tf

def train_step(vit, rag_head, retriever, optimizer, bce_loss,
               frames, metadata_batch, labels):

    with tf.GradientTape() as tape:
        vit_out = vit(frames, training=True)
        cls_embeddings = vit_out["pre_logits"]  # (B, 768)

        # Python retrieval step
        cls_np = cls_embeddings.numpy()
        retrieved_np = retriever(cls_np, metadata_batch)
        retrieved_embeddings = tf.convert_to_tensor(retrieved_np, dtype=tf.float32)

        logits, _ = rag_head(cls_embeddings, retrieved_embeddings, training=True)
        loss = bce_loss(labels, logits)

    train_vars = vit.trainable_variables + rag_head.trainable_variables
    grads = tape.gradient(loss, train_vars)
    optimizer.apply_gradients(zip(grads, train_vars))

    return loss