import spacy
import nltk
from nltk.corpus import words, brown
from collections import defaultdict
import numpy as np
import os
import pickle
from rapidfuzz import fuzz
import logging

class EnhancedLanguageProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.word_freq = defaultdict(int)
        self.ngram_model = defaultdict(lambda: defaultdict(float))
        self.current_word = []
        self.sentence = []
        self._ensure_nltk_data()
        self._load_resources()

    def _ensure_nltk_data(self):
        """Ensure required NLTK data is downloaded"""
        try:
            nltk.data.find('corpora/words')
        except LookupError:
            self.logger.info("Downloading NLTK words corpus...")
            nltk.download('words', quiet=True)

        try:
            nltk.data.find('corpora/brown')
        except LookupError:
            self.logger.info("Downloading NLTK brown corpus...")
            nltk.download('brown', quiet=True)

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            self.logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)

    def _load_resources(self):
        """Load dictionaries and pre-trained models"""
        try:
            # Load English words with frequencies
            self.logger.info("Loading word frequencies...")
            english_words = words.words()
            for word in english_words:
                self.word_freq[word.lower()] += 1

            # Add Brown corpus frequencies
            brown_words = brown.words()
            for word in brown_words:
                self.word_freq[word.lower()] += 1

            # Load pre-built ngram model if exists
            if os.path.exists("assets/models/ngram.pkl"):
                with open("assets/models/ngram.pkl", "rb") as f:
                    self.ngram_model = pickle.load(f)

            # Initialize spaCy
            self.logger.info("Loading spaCy model...")
            self.nlp = spacy.load("en_core_web_sm")

        except Exception as e:
            self.logger.error(f"Error loading resources: {str(e)}")
            raise

    def get_suggestions(self, partial_word, max_suggestions=5):
        """Get spelling corrections and completions"""
        try:
            # 1. Exact matches (prioritize common words)
            exact_matches = [
                w for w in self.word_freq
                if w.startswith(partial_word.lower())
            ]
            exact_matches.sort(key=lambda w: self.word_freq[w], reverse=True)

            # 2. Fuzzy matches
            if len(exact_matches) < max_suggestions:
                all_words = list(self.word_freq.keys())
                similarities = [
                    (w, fuzz.ratio(partial_word.lower(), w))
                    for w in all_words
                ]
                similarities.sort(key=lambda x: x[1], reverse=True)
                fuzzy_matches = [w for w, _ in similarities[:max_suggestions*2]]
                return list(set(exact_matches + fuzzy_matches))[:max_suggestions]

            return exact_matches[:max_suggestions]
        except Exception as e:
            self.logger.error(f"Error in get_suggestions: {str(e)}")
            return []

    def predict_next_word(self, previous_words, max_predictions=3):
        """Predict next word using n-gram model"""
        try:
            if not previous_words:
                return []

            context = " ".join(previous_words[-3:])  # Use trigram context
            doc = self.nlp(context)

            # Get POS-based predictions
            pos_pattern = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    pos_pattern.append(token.pos_)

            predictions = []
            for word in self.ngram_model[context]:
                score = self.ngram_model[context][word]

                # Check if word matches expected POS pattern
                word_doc = self.nlp(word)
                if len(pos_pattern) > 0 and word_doc[0].pos_ != pos_pattern[-1]:
                    score *= 0.7  # Penalize mismatches

                predictions.append((word, score))

            predictions.sort(key=lambda x: x[1], reverse=True)
            return [w for w, _ in predictions[:max_predictions]]
        except Exception as e:
            self.logger.error(f"Error in predict_next_word: {str(e)}")
            return []

    def update_model(self, sentence):
        """Update ngram frequencies from user input"""
        try:
            doc = self.nlp(sentence.lower())
            tokens = [token.text for token in doc if token.is_alpha]

            # Update ngrams
            for i in range(len(tokens) - 1):
                context = " ".join(tokens[max(0, i - 2):i + 1])
                next_word = tokens[i + 1]
                self.ngram_model[context][next_word] += 1

            # Periodic save
            if len(sentence.split()) % 100 == 0:
                self._save_model()
        except Exception as e:
            self.logger.error(f"Error in update_model: {str(e)}")

    def update_current_word(self, char):
        """Update the current word being built"""
        try:
            if char in [' ', '<space>']:
                if self.current_word:
                    self.sentence.append(''.join(self.current_word))
                    self.current_word = []
            elif char == '<del>' and self.current_word:
                self.current_word.pop()
            else:
                self.current_word.append(char)

            current = ''.join(self.current_word)
            return self.get_suggestions(current) if current else []
        except Exception as e:
            self.logger.error(f"Error in update_current_word: {str(e)}")
            return []

    def _save_model(self):
        """Save ngram model to disk"""
        try:
            os.makedirs("assets/models", exist_ok=True)
            with open("assets/models/ngram.pkl", "wb") as f:
                pickle.dump(self.ngram_model, f)
            self.logger.info("N-gram model saved successfully.")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
