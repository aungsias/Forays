import re

def process_text(text, lower_case, remove_stopwords, stop_words, regex_pattern="[^a-zA-Z\s]"):
    """
    Cleans and processes a given text.

    Parameters:
    - text (str): The text to be processed.
    - lower_case (bool): Whether to convert the text to lower case.
    - remove_stopwords (bool): Whether to remove stop words.
    - stop_words (list): A list of stop words to remove.
    - regex_pattern (str): A regex pattern to remove unwanted characters. Defaults to "^[a-zA-Z\s]".

    Returns:
    - str: The processed text.
    
    The function performs the following operations in sequence:
    1. Removes retweets, mentions, and hyperlinks.
    2. Optionally converts text to lower case.
    3. Removes unwanted characters based on regex pattern.
    4. Optionally removes stop words.
    5. Joins the words back into a string and returns it.
    """
    
    # Remove retweets, mentions, and links
    text = re.sub(r'^RT[\s]+', '', text)
    text = re.sub(r'\$\w*', '', text)
    text = re.sub(r'https?://[^\s\n\r]+', '', text)
    text = re.sub(r'https?://[A-Za-z0-9./]+','', text)
    text = re.sub(r'https//[A-Za-z0-9./]+','', text)
    text = text.replace("\n", "")
    text = re.sub('@[\w]+', '', text)

    # Handle contractions
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)

    # Convert to lower case if specified
    if lower_case:
        text = text.lower()

    # Remove characters based on regex pattern
    text = re.sub(regex_pattern, "", text)

    # Tokenize text
    word_list = text.split()
    word_list = [w for w in word_list if len(w) > 1]

    # Remove stop words if specified
    return " ".join([word for word in word_list if word not in stop_words]) if remove_stopwords \
        else " ".join(word_list)