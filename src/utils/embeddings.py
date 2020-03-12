import numpy as np
from src.utils.files import load_pickle_file


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


def load_sent_embeddings(source_file):
    """ Load sentence embeddings saved as pickle file

        :param source_file: The pickle file where the embeddings are stored
        :return: a list of sentence embeddings
    """
    sent_embeddings = load_pickle_file(source_file)

    sent_embed = []
    for img_id, embed in sent_embeddings.items():
        e = embed.numpy()
        sent_embed.append(e)
    sent_embed = np.array(sent_embed)
    return sent_embed


def load_embeds(text_embed, img_embed, dcca_embed):
    """ Load image and sentence embeddings and create a concatenated version

    :param text_embed: pickle file containing the sentence embeddings
    :param img_embed: pickle file containing the image embeddings
    :param dcca_embed: pickle file containing the deep cca embeddings
    :return: tuple containing sentence embeddings, image embeddings, concatenated embeddings
    """
    img_embed = load_img_embeddings(img_embed)
    sent_embed = load_sent_embeddings(text_embed)
    dcca_embed = load_pickle_file(dcca_embed)
    concat = np.concatenate((sent_embed, img_embed), axis=1)
    return sent_embed, img_embed, dcca_embed, concat


def retrieve_all_embeds(to_retrieve):
    """ Retrieve all embeddings of a list

    :param to_retrieve: a list containing tuples of (sentence_embeddings, image_embeddings)
    :return: a dictionary of list of image only, text only and concatenated embeddings.
    The embeddings in each list are retrieved in order (e.g. if the first tuple of to_compute is test dataset,
    test embeddings will be the first ones in each list)
    """
    embed = {"image only": [],
             "text only": [],
             "deep cca": [],
             "concatenated": []}

    for sent_embed_file, img_embed_file, dcca_embed_file in to_retrieve:
        sent_embed, img_embed, dcca_embed, concat_embed = load_embeds(sent_embed_file, img_embed_file, dcca_embed_file)
        embed["image only"].append(img_embed)
        embed["text only"].append(sent_embed)
        embed["deep cca"].append(dcca_embed)
        embed["concatenated"].append(concat_embed)

    return embed
