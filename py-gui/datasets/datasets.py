import os
import pandas as pd
import re

# Dictionary mapping of emoticons to sentiment words
EMOTICON_MAPPING = {
    # standard emoticons
    ":)": "happy",
    ":D": "happy joyful",
    ":(": "sad",
    ":/": "sad confused",
    ":|": "neutral",
    ":*": "happy kiss",
    ":P": "happy playful",
    ":O": "sad surprised",
    ":@": "sad angry",
    ":S": "sad worried",
    ":$": "sad embarrassed",
    ":X": "neutral sealed lips",
    ":#": "neutral mute",
    ":&": "sad tired",
    ":!": "sad surprised",
    ":?": "sad confused",
    ":;": "happy wink",
    ":.": "sad confused",
    ":,": "sad confused",
    ":^": "sad confused",
    ":'(": "sad",
    ":')": "happy",
    ":/": "confused",
    ":\\": "confused",
    # throw up/ sick/ barf emoticon
    ":&(": "sad sick angry",
    ":&-(": "sad sick angry",
    ":&-[": "sad sick angry",
    ":&[": "sad sick angry",
    ":&{": "sad sick angry",
    # heart emoticons
    "<3": "happy love",
    "</3": "broken heart, sad love",
    "<\\3": "broken heart, sad love",
    "<\\": "broken heart, sad love",
    # face palm emoticons
    ":facepalm:": "embarrassed",
    ":fp:": "embarrassed",
    ":face_palm:": "embarrassed",
}

def replace_emoticons_with_words(text):
    for emoticon, word in EMOTICON_MAPPING.items():
        text = re.sub(re.escape(emoticon), word, text)
    return text

def clean_text(text):
    # remove links
    text = re.sub(r'http\S+', '', text)
    # remove hashtags
    text = re.sub(r'#\S+', '', text)
    # remove mentions
    text = re.sub(r'@\S+', '', text)
    # remove newline characters
    text = re.sub(r'\n', '', text)
    # remove HTML references
    text = re.sub(r'&\S+', '', text)
    # substitute emoticons with words
    text = replace_emoticons_with_words(text)
    # remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # remove remaining  punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # convert to lowercase
    text = text.lower()
    return text


def clean_and_parse_df(df, text_column='tweet'):
    df['cleaned_text'] = df[text_column].apply(clean_text)
    return df


