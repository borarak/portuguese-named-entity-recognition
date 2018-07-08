# The trained MITIE corpora vectors
CORPORA_VECTOR = "/data/total_word_feature_extractor.dat"

# NLTK portuguese sentence tokenizer
SENTENCE_TOKENIZER_POR = "tokenizers/punkt/portuguese.pickle"

# the annotated files you want to train on
TRAIN_FILES_DIR = "/data/annotated/3rd_batch/all"

# The files you want to evaluate on
TEST_FILES_DIR = "/data/test/"

# List of all valid annotations go below (e.g Person)
ANNOTATIONS =[]

# the trained NER model location
MODEL_FILE = "/models/ner_model_v7_bran_aws.dat"


# Number of CPU cores for training, more = quicker traning process
TRAINING_CPU_THREADS = 6

# Log directory
LOG_DIR = "/logs"

# place where NER predictions are saved
TEST_RESULTS_FILE = "/results/predictions.txt"

# Place you BRAT annotations here for post_procesing
# Files must end with ".txt" and have a corresponding ".ann" file
BRAT_ANNOTATIONS_DIR = "/data/annotated/"

# Locations where post-processed BRAT annotations are saved
BRAT_POSTPROCESSED_DIR = "/data/postprocessed/"


# Confidence scores
PERSON_SCORE = 1.2
COMPANY_SCORE = 2.0
CNPJ_SCORE = 1.0
CPF_SCORE = 0.5
SHARES_SCORE = 0.1
SHARES_VALUE_SCORE = 0.7