import logging

import pandas as pd
from src.utils.files import save_as_pickle

logger = logging.getLogger("text_feature_extractor")
logger.setLevel(logging.INFO)


class TextFeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.sents = None
        self.img_ids = None

    def extract_sentences_and_ids(self, dataset_path):
        df = pd.read_csv(dataset_path)
        self.sents = df["Corrected_text"].astype("str")
        self.img_ids = df["image_name"].astype("str")

    def extract_all_features(self, dataset_path, outfile=None):
        self.extract_sentences_and_ids(dataset_path)
        embeddings = self.model(self.sents)
        features = {}
        for i, (embed, sent, img_id) in enumerate(zip(embeddings, self.sents, self.img_ids)):
            logger.debug("{}\t{}\t\t{}".format(i, sent, img_id))
            features[img_id] = embed

        if outfile is not None:
            save_as_pickle(features, outfile)

        return features
