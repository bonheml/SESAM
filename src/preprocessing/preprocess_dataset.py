import numpy as np
from src.preprocessing.image_preprocessing import ImageFeatureExtractor
import tensorflow_hub as hub
from pathlib import Path
from src.preprocessing.text_preprocessing import TextFeatureExtractor


def extract_all_images(model_name, images_directory, features_outfile=None):
    """ Extract features from pictures and save them in a pickle file

    :param model_name: the name of the model to use
    :param images_directory: folder where the images are stored
    :param features_outfile: file used to store the extracted features
    :return: None
    """
    feature_extractor = ImageFeatureExtractor(model_name)
    return feature_extractor.extract_all_features(images_directory, features_outfile)


def extract_all_sentences(dataset_path, features_outfile=None):
    """ Extract features from sentences using pretrained universal sentence embeddings and save them in a pickle file

    :param dataset_path: the path of the dataset to use
    :param features_outfile: file used to store the extracted features
    :return: extracted embeddings
    """
    model_path = Path(__file__).parent.parent.parent / "data" / "models" / "use"
    use = hub.load(str(model_path.absolute()))
    feature_extractor = TextFeatureExtractor(use)
    return feature_extractor.extract_all_features(dataset_path, features_outfile)


def df_to_embed(df, img_folder):
    """ Extract image embeddings, sentence embeddings and concatenated embeddings from dataset and image folders

    :param df: dataset file to use
    :param img_folder: folder where the corresponding images are stored
    :return: tuple containing sentence embeddings, image embeddings, concatenated embeddings
    """
    sent_embed = extract_all_sentences(df)
    img_embed = extract_all_images("xception", img_folder)
    concat = np.concatenate((sent_embed, img_embed), axis=1)
    return sent_embed, img_embed, concat


def compute_all_embeds(to_compute):
    """ Compute all embeddings of a list

    :param to_compute: a list containing tuples of (dataset_file, image_folder)
    :return: a dictionary of list of image only, text only and concatenated embeddings.
    The embeddings in each list are extracted in order (e.g. if the first tuple of to_compute is test dataset,
    test embeddings will be the first ones in each list)
    """
    embed = {"image only": [],
             "text only": [],
             "concatenated": []}

    for df, img_folder in to_compute:
        sent_embed, img_embed, concat_embed = df_to_embed(df, img_folder)
        embed["image only"].append(img_embed)
        embed["text only"].append(sent_embed)
        embed["concatenated"].append(concat_embed)

    return embed

