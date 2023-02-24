from lab_utils import LabPredictor
import spacy
import nltk 
from .models import TrigramModel
import re
from nltk.collocations import *

# pylint: disable=pointless-string-statement
"""
Second Lab! POS tagging with spaCy.

- As for Lab 1, it's up to you to change anything with the structure, 
    as long as you keep the class name (Lab2, inheriting from LabPredictor)
    and the methods (predict, train)


While NLTK has pre-tagged corpora available, I want you to use spaCy to
compute POS tags on just the text (same as Lab 1),
which is more realistic for a real-world application.

An important note:
- Can you compute the POS tag from a single token, or do you need to look at the context (i.e. sentence)?

"""

class Lab2(LabPredictor):
    def __init__(self):
        super().__init__()
        self.corpus = " ".join(nltk.corpus.brown.words(categories="news"))
        # TODO: select a strategy for cold start (when missing words
        self.start_words = ["the place", "i went", "where is", "this is", "that is", "these are", "those are"]
        self.model = None  # the model will be loaded/made in `train`
        self.nlp = None
        self.doc = None
    
    def preprocess(sent):
        sent = " ".join(sent)
        sent = re.sub(r"[^\w,.!?]", " ", sent)
        sent = re.sub(r"\s+", " ", sent)
        return sent.strip()

    def predict(self, text):
        print(f"Lab2 receiving: {text}")

        lastBigram = text.split()[-2:]

        predictions = []

        # TODO: apply your own idea of using POS tags to alter/filter the predictions
        # you can also implement anything you wish from spacy.
        # there's a lot of interesting stuff in the spacy docs.

        # Morphological Analysis
        trigram_measures = nltk.collocations.TrigramAssocMeasures()
        nlpedinput = self.nlp(" ".join(list(text)))
        inputtags = []
        for token in nlpedinput:
            inputtags.append(token.tag_)
        tags = []
        texts = []
        for token in doc:
            tags.append(token.tag_)
            tags.append(token.text)

        finder = TrigramCollocationFinder.from_words(tags)
        findertexts = TrigramCollocationFinder.from_words(texts)
        scored = finder.score_ngrams(trigram_measures.raw_freq)
        scoredtexts = findertexts.score_ngrams(trigram_measures.raw_freq)
        alllist = []
        for tuple1,tuple2 in zip(scored,scoredtexts):
            alllist.append((tuple1, tuple2))
        print(alllist[0:50])
        good_tag = ""
        good_trigram = ()
        for trigram in scored:
            trigramlist = list(trigram)
            if((trigramlist[0],trigramlist[1]) == bigram_to_predict):
                good_tag = trigramlist[2]
                good_trigram = trigram
        filter_trigrams_from_words = []
        finderWords = TrigramCollocationFinder.from_words(tags)
    

        scored = finder.score_ngrams(trigram_measures.raw_freq)
        best = sorted(finder.nbest(trigram_measures.raw_freq, 2))
    
        
        
        return self.model.predict(text, n_words=4)

    def train(self) -> None:
        # TODO: use the trigram model from Lab 1 or the one provided in the solutions folder
        # TODO: NEW TO LAB 2: load spacy model for POS tagging
        self.nlp = spacy.load("en_core_web_sm")
        self.doc = list(self.nlp(self.corpus, exclude=["tagger", "tok2vec"]))
        self.model = TrigramModel(docs)
        
