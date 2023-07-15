import json


DATASET_PATH = '/content/gdrive/MyDrive/dls_project'
ANNS_FILE_PATH = DATASET_PATH + '/' + 'annotations.json'

with open(ANNS_FILE_PATH, 'r') as f:
    dataset = json.loads(f.read())

categories = dataset['categories']

LABELS = {}
for cat in categories:
    LABELS[cat['supercategory']] = []
for cat in categories:
    LABELS[cat['supercategory']].append(cat['id'])

config = dict(
    DATASET_PATH=DATASET_PATH,
    ANNS_FILE_PATH=ANNS_FILE_PATH,
    LABELS=LABELS
)
