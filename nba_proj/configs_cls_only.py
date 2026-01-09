import os 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_store")
# rag_head_7_vids_bs4_cls_only_12_epochs
RAG_WEIGHTS = "rag_weights/rag_head_7_vids_bs4_cls_only_12_epochs_rebuild.weights.h5"
PROJ_WEIGHTS = 'proj_weights/projection_7_vids_bs4_cls_only_12_epochs_rebuild.weights.h5'

CHROMADB_COLLECTION = "rich_embeddings_7_vids_bs4_cls_only_12_epochs_rebuild"

TOP_K = 100
SEARCH_K = 750

EPOCHS = 12
REBUILD_EVERY = 3
ACCUM_BATCH_SIZE = 8

ADJUST_CONTRASTIVE_LOSS_EVERY = 2
INCREMENT_CONTRASTIVE_LOSS_BY = 0.025
# PHASE_1_CONTRASTIVE_LOSS = 0.0
# PHASE_2_CONTRASTIVE_LOSS = 0.1

PHASE_1_LEARNING_RATE = 1e-5
PHASE_2_LEARNING_RATE = 1e-6

NUM_QUERIES = 16
NUM_LAYERS = 4
NUM_HEADS = 8

START_CHUNK_TRAIN = 0
END_CHUNK_TRAIN = 3000

START_CHUNK_VALID = 3100
END_CHUNK_VALID = 3300

CHUNK_BATCH_SIZE = 4
PRINT_EVERY = 25

NUM_CLIPS_PER_VID = 15
# VIDS_TO_USE = ['vid3','vid4','vid6']
VIDS_TO_USE = ['vid1',"vid2", 'vid3','vid4','vid6','vid8','vid10']
# VIDS_TO_USE = ['vid3']