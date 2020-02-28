from os import listdir
from os.path import abspath
from tensorflow.keras import Model
from tensorflow.keras.applications import Xception, VGG16, VGG19
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from src.utils.files import save_as_pickle
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageFeatureExtractor:
    def __init__(self, model_name='xception'):
        self.model = self._prepare_model(model_name)
        if model_name is 'xception':
            self.input_shape = (299, 299)
        else:
            self.input_shape = (224, 224)

    def _prepare_model(self, model_name):
        """ Initialise the feature extractor Here we remove the prediction layer in order to perform the feature
        extraction only.
        :param model_name: name of the model
        :return: the model with last layer removed
        """
        models = {"xception": Xception,
                  "VGG16": VGG16,
                  "VGG19": VGG19}
        if model_name not in models:
            raise NotImplementedError
        model = models[model_name]()
        model.layers.pop()
        model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
        return model

    def extract_features(self, input_file):
        """ Load image from file 'input_file' convert it to an array and reshape it to use it with Xception model
        compute the intern pre processing needed by Xception and predict the features of the image
        :param input_file: the image file
        :return: the features extracted
        """
        filename = abspath(input_file)
        img = load_img(filename, target_size=self.input_shape)
        vectorized_img = img_to_array(img)
        vectorized_img = vectorized_img.reshape((1, vectorized_img.shape[0],
                                                 vectorized_img.shape[1],
                                                 vectorized_img.shape[2]))
        preprocessed_img = preprocess_input(vectorized_img)
        features = self.model.predict(preprocessed_img, verbose=0)
        return features

    def extract_all_features(self, directory, outfile=None):
        """ Extract features from all images in the folder 'directory'
        :param directory: image folder
        :param outfile: Optional filename used to save the features. If None no save will be performed.
        :return: dict of features
        """
        features = {}

        for file in listdir(directory):
            file_path = "/".join([directory, file])
            features[file] = self.extract_features(file_path)

        if outfile is not None:
            save_as_pickle(features, outfile)

        return features
