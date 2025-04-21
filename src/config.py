from calendar import EPOCH


ORIGINAL_DATASET_PATH = "movie_amz.json"
AMZ_DATA_PATH = "dataset/amazon"
AMZ_CLOTHING_DATA_PATH = "dataset/amazon_clothing"
ROTTEN_DATA_PATH = "dataset/rotten"


## train 
LOADER_WORKERS = 8
BATCH_SIZE = 128
EPOCH = 10
DEVICE = 'cuda:0'

TRAIN_RATIO = 0.6
VALID_RATIO = 0.2

# model 
EDGE_FRATURES_DIM = 768

LATENT_DIMS = [32,32,32,32]
NUM_RATING = 10

LEARING_RATE = 0.02
WEIGHT_DECAY = 0.9