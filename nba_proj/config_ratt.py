import os 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_store")
# rag_head_7_vids_bs4_cls_only_12_epochs
RATT_WEIGHTS = "rag_weights/ratt_w.weights.h5"
PROJ_WEIGHTS = 'proj_weights/ratt_proj.weights.h5'

CHROMADB_COLLECTION = "ratt_db"

TOP_K = 5
SEARCH_K = 750

EPOCHS = 12
REBUILD_EVERY = 3
ACCUM_BATCH_SIZE = 1

ADJUST_CONTRASTIVE_LOSS_EVERY = 2
INCREMENT_CONTRASTIVE_LOSS_BY = 0.025
# PHASE_1_CONTRASTIVE_LOSS = 0.0
# PHASE_2_CONTRASTIVE_LOSS = 0.1

PHASE_1_LEARNING_RATE = 1e-5
PHASE_2_LEARNING_RATE = 1e-6

NUM_QUERIES = 16
NUM_LAYERS = 4
NUM_HEADS = 8

CHUNK_SIZE = 12

START_CHUNK_TRAIN = 200
END_CHUNK_TRAIN = 300

START_CHUNK_VALID = 310
END_CHUNK_VALID = 330

CHUNK_BATCH_SIZE = 4
PRINT_EVERY = 5

NUM_CLIPS_PER_VID = 3
# VIDS_TO_USE = ['vid3','vid4','vid6']
VIDS_TO_USE = ['vid1',"vid2", 'vid3','vid4','vid6','vid8','vid10']
# VIDS_TO_USE = ['vid3']