import os 
from datetime import datetime
import uuid


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_store")
# rag_head_7_vids_bs4_cls_only_12_epochs

# ratt_w_ordered_new 4 toks
# ratt_w_ordered_new_32toj 32 tokens
CHROMADB_COLLECTION = "ratt_db_chunk_encoder_all_vids_relcls_new" #ratt_db_chunk_encoder_all_vids_relcls_v2
# ratt_db_chunk_encoder_all_vids_v2
# ratt_db_cached_30_clips_ordered_fixed_rebuild_w_labels v26

DELTA_T_NORM = 0.05 #this was 0.05

STAGE1_WEIGHTS = './chunk_encoder_ckpts_cached/chunk_encoder_best.weights.h5'
BUILD_RET_C = 75 #this was a 100
TOP_K = 32
SEARCH_K = 100#750

EPOCHS = 12
REBUILD_EVERY = 3
ACCUM_BATCH_SIZE = 1

ADJUST_CONTRASTIVE_LOSS_EVERY = 2
# INCREMENT_CONTRASTIVE_LOSS_BY = 0.025
INCREMENT_CONTRASTIVE_LOSS_BY = 0.025
# PHASE_1_CONTRASTIVE_LOSS = 0.0
# PHASE_2_CONTRASTIVE_LOSS = 0.1

PHASE_1_LEARNING_RATE = 1e-3 # was -5 and then -6
PHASE_2_LEARNING_RATE = 1e-3

CACHE_PATH = f"./cache/retrieval_cache_{CHROMADB_COLLECTION}_searchk_{SEARCH_K}_chunk_encoded_new.pkl" #temporary_new_way
STAGE3_CACHE_PATH = f"./cache/stage3_retrieval_cache_{CHROMADB_COLLECTION}_searchk_{SEARCH_K}_chunk_encoded_new.pkl"

NUM_QUERIES = TOP_K #was 16
NUM_LAYERS = 12
NUM_HEADS = 8

CHUNK_SIZE = 12

START_CHUNK_TRAIN = 0
END_CHUNK_TRAIN = 3750

START_CHUNK_VALID = 4000
END_CHUNK_VALID = 4100

CHUNK_BATCH_SIZE = 16
PRINT_EVERY = 1

NUM_CLIPS_PER_VID = 30
# VIDS_TO_USE = ['vid3','vid4','vid6']
VIDS_TO_USE = ["vid2", 'vid3','vid4','vid5','vid6','vid7','vid8','vid10']
TRAIN_VIDS = ['vid2','vid3','vid4','vid5','vid6','vid7','vid8']
TEST_VIDS = ['vid10']


RUN_STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_UID = uuid.uuid4().hex[:4]

RUN_ID = (
    f"{RUN_STAMP}_"
    f"vtest-{'-'.join(TEST_VIDS)}_"
    f"db-{CHROMADB_COLLECTION}_"
    f"ret{BUILD_RET_C}_k{TOP_K}_sk{SEARCH_K}_"
    f"dt{str(DELTA_T_NORM).replace('.', '')}_"
    f"ch{CHUNK_SIZE}_L{NUM_LAYERS}H{NUM_HEADS}Q{NUM_QUERIES}_"
    f"bs{CHUNK_BATCH_SIZE}_acc{ACCUM_BATCH_SIZE}_"
    f"e{EPOCHS}_lr{PHASE_1_LEARNING_RATE:.0e}to{PHASE_2_LEARNING_RATE:.0e}_"
    f"reb{REBUILD_EVERY}_"
    f"{RUN_UID}"
)

RATT_WEIGHTS = f"rag_weights/{RUN_ID}.weights.h5" 
PROJ_WEIGHTS = 'proj_weights/ratt_proj_ordered.weights.h5' 

# VIDS_TO_USE = ['vid3']