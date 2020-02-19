from os.path import splitext
from pickle import load, dump
import numpy as np


def load_pickle_file(filename):
    """ Load pickle file containing features
    :param filename: Name of the pickle file to load
    :return: loaded features
    """
    with open(filename, mode='rb') as f:
        features = load(f)
    return features


def save_as_pickle(to_save, outfile):
    """ Save data as pickle file
    :param to_save: data to save
    :param outfile: filename of the pickle file
    :return: None
    """
    with open(outfile, 'wb') as out:
        dump(to_save, out)


def get_image_ids(filename):
    """
    Get image ids from a file and remove the jpg extension
    :param filename: filename of the file containing the image ids
    :return: dict of image ids
    """
    with open(filename) as f:
        image_ids = [splitext(i)[0] for i in f.read().split('\n') if i]
    return image_ids


def load_img_embeddings(source_file):
    """ Load image embeddings saved as pickle file

    :param source_file: The pickle file where the embeddings are stored
    :return: a list of image embeddings
    """
    img_embeddings = load_pickle_file(source_file)
    img_embed = list(img_embeddings.values())
    img_embed = np.array(img_embed)
    img_embed = img_embed.reshape((img_embed.shape[0], img_embed.shape[2]))
    return img_embed


def load_sent_embeddings(source_file, df):
    """ Load sentence embeddings saved as pickle file

        :param source_file: The pickle file where the embeddings are stored
        :param df: the dataset source (as pandas dataframe)
        :return: a list of sentence embeddings
    """
    sent_embeddings = load_pickle_file(source_file)

    # For an unknown reason mapping numpy conversion to all tensors with list.map or during pickle save
    # remove some of the entries so I use loop instead to prevent entries removal
    img_ids = df["image_name"]
    sent_embed = []
    for i, img_id in enumerate(img_ids):
        e = sent_embeddings[img_ids[i]].numpy()
        sent_embed.append(e)
    sent_embed = np.array(sent_embed)
    return sent_embed

