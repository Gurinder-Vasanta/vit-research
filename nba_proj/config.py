RAG_WEIGHTS = "rag_weights/rag_head_test_vids_bs2.weights.h5"
PROJ_WEIGHTS = "proj_weights/projection_head_test_vids_bs2.weights.h5"

CHROMADB_COLLECTION = "rich_embeddings_test_vids_bs2"

TOP_K = 10
SEARCH_K = 750

EPOCHS = 24
REBUILD_EVERY = 4
ACCUM_BATCH_SIZE = 8

PHASE_1_CONTRASTIVE_LOSS = 0.0
PHASE_2_CONTRASTIVE_LOSS = 0.1

PHASE_1_LEARNING_RATE = 1e-4
PHASE_2_LEARNING_RATE = 1e-6

NUM_QUERIES = 16
NUM_LAYERS = 6
NUM_HEADS = 8

START_CHUNK_TRAIN = 0
END_CHUNK_TRAIN = 100

START_CHUNK_VALID = 120
END_CHUNK_VALID = 130

CHUNK_BATCH_SIZE = 6
PRINT_EVERY = 5

NUM_CLIPS_PER_VID = 2
VIDS_TO_USE = ['vid3','vid4','vid6']
# VIDS_TO_USE = ['vid1',"vid2", 'vid3','vid4','vid6','vid8','vid10']
# VIDS_TO_USE = ['vid3']