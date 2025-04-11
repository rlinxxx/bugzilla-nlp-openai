import re
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from krovetzstemmer import Stemmer 

class PreProcessor:

  def __init__(self, texts = None):
    self.texts = texts
  
  def remove_chars(self):
    regex_list = ["http\S+", 
                "[-|a-z|0-9\.]+\.com",
                "\([^)]*\)", 
                "\[[^]]*\]", 
                "\*[^*]*\*", 
                "\{[^}]*\}", 
                "\d+", 
                "[^\w\s]", 
                "[^\x00-\x7F]",
                "_",
                " +",
                "(\n)",
                "(\r)"]
    regex = re.compile(r"|".join(regex_list))
    self.texts = regex.sub(" ", self.texts) 
    return self

  def get_words(self):
    token = tokenize.WhitespaceTokenizer()
    self.words = token.tokenize(self.texts)
    return self

  def remove_stopwords(self):
    stoplist = set(stopwords.words('english')) - set(['no', 'not', 'off', 'on', 'don', 'down', 'above', 'over', 
                                                    'under', 'after', 'below', 'until', 'out', 'can', 'does', 'don', 'doesn'])
    processed_text = list()
    for word in self.words:
      if word not in stoplist and len(word) > 1:
        processed_text.append(word)
    self.words = processed_text
    return self

  def remove_special_stopwords():
    pass

  def remove_lemma(self):
    wnl = WordNetLemmatizer()
    processed_text = list()
    for word in self.words:
      processed_text.append(wnl.lemmatize(word))
    self.words = processed_text
    return self

  def remove_stem(self):
    stemmer = Stemmer()
    processed_text = list()
    for word in self.words:
      processed_text.append(stemmer.stem(word))
    self.words = processed_text
    return self

  def join_words(self):
    self.words = ' '.join(self.words)
    return self

  def clean_text(self, texts):
    self.texts = texts
    self = self.remove_chars()
    self = self.get_words()
    self = self.remove_stopwords()
    self = self.remove_lemma()
    self = self.remove_stem()
    self = self.join_words()
    return self.words
