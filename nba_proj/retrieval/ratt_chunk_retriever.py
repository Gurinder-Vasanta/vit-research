# import numpy as np

# class RattChunkRetriever:
#     def __init__(self, collection, top_k=10, search_k=200):
#         self.collection = collection
#         self.top_k = top_k
#         self.search_k = search_k

#     def __call__(self, chunk_embs, metadata):
#         if hasattr(chunk_embs, "numpy"):
#             q_np = chunk_embs.numpy()
#         else:
#             q_np = chunk_embs

#         B = q_np.shape[0]
#         retrieved_all = []

#         for i in range(B):
#             vid      = int(metadata["vid"][i].numpy())
#             side     = metadata["side"][i].numpy().decode()
#             t_center = float(metadata["t_center"][i].numpy())
#             t_width  = float(metadata["t_width"][i].numpy())

#             q_start = t_center - t_width / 2
#             q_end   = t_center + t_width / 2

#             result = self.collection.query(
#                 query_embeddings=[q_np[i].tolist()],
#                 n_results=self.search_k,
#                 where={
#                     "$and": [
#                         {"vid_num": {"$ne": vid}},
#                         {"side": side},
#                         {"t_center": {"$gte": q_start}},
#                         {"t_center": {"$lte": q_end}},
#                     ]
#                 },
#                 include=["embeddings",'metadatas']
#             )

#             retrieved_vecs = result["embeddings"][0]

#             if len(retrieved_vecs) >= self.top_k:
#                 retrieved_vecs = retrieved_vecs[:self.top_k]
#             else:
#                 D = q_np.shape[1]
#                 pad = np.zeros((self.top_k - len(retrieved_vecs), D), dtype=np.float32)
#                 retrieved_vecs = np.vstack([retrieved_vecs, pad])

#             retrieved_vecs = np.array(retrieved_vecs, dtype=np.float32)
#             retrieved_vecs /= (np.linalg.norm(retrieved_vecs, axis=1, keepdims=True) + 1e-8)

#             retrieved_all.append(retrieved_vecs)

#         return np.stack(retrieved_all, axis=0)




import numpy as np

class RattChunkRetriever:
    def __init__(self, collection, top_k=10, search_k=200):
        self.collection = collection
        self.top_k = top_k
        self.search_k = search_k

    def __call__(self, chunk_embs, metadata):
        if hasattr(chunk_embs, "numpy"):
            q_np = chunk_embs.numpy()
        else:
            q_np = chunk_embs

        q_np = np.asarray(q_np, dtype=np.float32)
        q_np /= (np.linalg.norm(q_np, axis=1, keepdims=True) + 1e-8)

        B, D = q_np.shape
        retrieved_all = np.zeros((B, self.top_k, D), dtype=np.float32)

        vids = np.array([int(metadata["vid"][i].numpy()) for i in range(B)], dtype=np.int32)
        sides = [metadata["side"][i].numpy().decode() for i in range(B)]
        t_centers = np.array([float(metadata["t_center"][i].numpy()) for i in range(B)], dtype=np.float32)
        t_widths = np.array([float(metadata["t_width"][i].numpy()) for i in range(B)], dtype=np.float32)

        unique_sides = sorted(set(sides))

        for side in unique_sides:
            idxs = [i for i in range(B) if sides[i] == side]
            if not idxs:
                continue

            q_starts = t_centers[idxs] - t_widths[idxs] / 2.0
            q_ends   = t_centers[idxs] + t_widths[idxs] / 2.0

            global_start = float(np.min(q_starts))
            global_end   = float(np.max(q_ends))

            result = self.collection.get(
                where={
                    "$and": [
                        {"side": side},
                        {"t_center": {"$gte": global_start}},
                        {"t_center": {"$lte": global_end}},
                    ]
                },
                include=["embeddings", "metadatas"]
            )

            cand_embs = result.get("embeddings", [])
            cand_meta = result.get("metadatas", [])

            if cand_embs is None or len(cand_embs) == 0:
                continue

            cand_embs = np.asarray(cand_embs, dtype=np.float32)
            cand_embs /= (np.linalg.norm(cand_embs, axis=1, keepdims=True) + 1e-8)

            cand_vids = np.array([int(m["vid_num"]) for m in cand_meta], dtype=np.int32)
            cand_tcenters = np.array([float(m["t_center"]) for m in cand_meta], dtype=np.float32)

            # Precompute similarities for all queries on this side against all side candidates
            # shape: (num_side_queries, num_candidates)
            sims = q_np[idxs] @ cand_embs.T

            for local_j, i in enumerate(idxs):
                q_start = t_centers[i] - t_widths[i] / 2.0
                q_end   = t_centers[i] + t_widths[i] / 2.0
                vid     = vids[i]

                valid_mask = (
                    (cand_vids != vid) &
                    (cand_tcenters >= q_start) &
                    (cand_tcenters <= q_end)
                )

                valid_idx = np.where(valid_mask)[0]

                if len(valid_idx) == 0:
                    continue

                valid_sims = sims[local_j, valid_idx]

                # take top_k best local matches
                if len(valid_idx) > self.top_k:
                    top_local = np.argpartition(-valid_sims, self.top_k - 1)[:self.top_k]
                    top_local = top_local[np.argsort(-valid_sims[top_local])]
                else:
                    top_local = np.argsort(-valid_sims)

                chosen_idx = valid_idx[top_local]
                retrieved_vecs = cand_embs[chosen_idx]

                if len(retrieved_vecs) < self.top_k:
                    pad = np.zeros((self.top_k - len(retrieved_vecs), D), dtype=np.float32)
                    retrieved_vecs = np.vstack([retrieved_vecs, pad])

                retrieved_all[i] = retrieved_vecs

        return retrieved_all

