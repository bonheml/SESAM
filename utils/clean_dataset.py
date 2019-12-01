import re
import logging

logger = logging.getLogger("dataset_cleaner")
logger.setLevel(logging.INFO)

special_quotes = re.compile(r"[＂〃ˮײ᳓″״‶˶ʺ“”˝‟]+", re.M)
unclosed_quotes = re.compile(r"(,([^\",]|(\"[^\",]*\"))*)(\")([^\",]*)(?=,)", re.M)


def remove_unclosed_quotes(content):
    m = unclosed_quotes.findall(content, re.M)
    logger.info("{} unclosed quotes found".format(len(m)))
    logger.debug(m)

    quotes_cleaned = re.sub(unclosed_quotes, r"\1\3", content)
    m = unclosed_quotes.findall(quotes_cleaned, re.M)
    logger.info("{} unclosed quotes found after replacement".format(len(m)))
    logger.debug(m)
    return quotes_cleaned


def replace_special_quotes(content):
    m = special_quotes.findall(content, re.M)
    logger.info("{} special quotes found".format(len(m)))
    logger.debug(m)

    quotes_replaced = re.sub(special_quotes, "\"", content)
    m = special_quotes.findall(quotes_replaced, re.M)
    logger.info("{} special quotes found after replacement".format(len(m)))
    logger.debug(m)
    return quotes_replaced


def cleanup_dataset(input_file, output_file):
    with open(input_file) as f_in:
        content = f_in.read()
    content_special_quotes_removed = replace_special_quotes(content)
    content_cleaned = remove_unclosed_quotes(content_special_quotes_removed)
    with open(output_file, "w+") as f_out:
        f_out.write(content_cleaned)
