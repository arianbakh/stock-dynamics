import os


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DEBUG_DIR = os.path.join(BASE_DIR, 'debug')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
XI_PATH = os.path.join(OUTPUT_DIR, 'xi.npy')
