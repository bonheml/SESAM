from src.baseline.image_preprocessing import FeatureExtractor


def extract_all(model_name, images_directory, features_outfile):
    """ Extract features from pictures and save them in a pickle file

    :param model_name: the name of the model to use
    :param images_directory: folder where the images are stored
    :param features_outfile: file used to store the extracted features
    :return: None
    """
    feature_extractor = FeatureExtractor(model_name)
    feature_extractor.extract_all_features(images_directory, features_outfile)

