import tensorflow as tf
from models.projection_head import ProjectionHead

INPUT_DIM = 768      # pooled ViT chunk embedding
HIDDEN = 768
PROJ_DIM = 768

# Build model
proj = ProjectionHead(input_dim=INPUT_DIM,
                      hidden_dim=HIDDEN,
                      proj_dim=PROJ_DIM)

proj.build((None, INPUT_DIM))

# Save initial weights
proj.save_weights("projection_head.weights.h5")
# proj.save_weights("projection_head_initial.h5")

# print("Initialized 768→768 projection head and saved as projection_head.h5")