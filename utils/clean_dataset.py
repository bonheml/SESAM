import re
import logging

logger = logging.getLogger("dataset_cleaner")
logger.setLevel(logging.INFO)

special_quotes = re.compile(r"[＂〃ˮײ᳓″״‶˶ʺ“”˝‟]+", re.M)


def replace_special_quotes(content):
    m = special_quotes.findall(content, re.M)
    logger.info("{} special quotes found".format(len(m)))
    logger.debug(m)

    quotes_replaced = re.sub(special_quotes, "", content)
    m = special_quotes.findall(quotes_replaced, re.M)
    logger.info("{} special quotes found after replacement".format(len(m)))
    logger.debug(m)
    return quotes_replaced


def cleanup_dataset(input_file, output_file):
    with open(input_file) as f_in:
        content = f_in.read()
    content_cleaned = replace_special_quotes(content)
    with open(output_file, "w+") as f_out:
        f_out.write(content_cleaned)
