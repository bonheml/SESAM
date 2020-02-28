from pickle import load, dump
import pandas as pd


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


def load_cat_df(dataset_path, use_cols, categorical_cols):
    """ Load dataframe with categorical columns

    :param dataset_path: the dataset to load (must be csv)
    :param use_cols: the name of the columns to use
    :param categorical_cols: list of tuples containing for each categorical column the name of the column, the name of
    the categories it contains and if the categories are ordered or not. e.g.: (col_name, cat_names, is_ordered)
    :return: The loaded pandas dataframe
    """
    df = pd.read_csv(dataset_path, usecols=use_cols)
    for col, categories, ordered in categorical_cols:
        df[col] = pd.Categorical(df[col], categories=categories, ordered=ordered)
    return df


def load_dfs(dfs):
    categories = [("Overall_sentiment", ["negative", "neutral", "positive"], True),
                  ("Humour", ["not_funny", "funny", "very_funny", "hilarious"], True),
                  ("Sarcasm", ["not_sarcastic", "general", "twisted_meaning", "very_twisted"], True),
                  ("Offense", ["not_offensive", "slight", "very_offensive", "hateful_offensive"], True),
                  ("Motivation", ["not_motivational", "motivational"], True)]
    use_cols = ["image_name", "Humour", "Sarcasm", "Offense", "Motivation", "Overall_sentiment", "Humour_bin",
                "Sarcasm_bin", "Offense_bin", "Motivation_bin"]

    return [load_cat_df(df, use_cols, categories) for df in dfs]


