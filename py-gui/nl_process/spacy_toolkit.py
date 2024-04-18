#
# nlp toolkit using the spacy library
# references: https://conference.scipy.org/proceedings/scipy2021/pdfs/jyotika_singh.pdf
# https://spacy.io/usage/spacy-101
#

import os
import pandas as pd
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from spacy import displacy

# write a class that contains functions for pre-processing and tokenizing text. The text will be in a pandas dataframe,
# and the functions will add a new column with the pre-processed text.
# If the dataframe contains a column called cleaned_text, the functions will use that column as input.

class SpacyToolkit:                
            def __init__(self, df, text_column='text'):
                self.df = df
                self.text_column = text_column
        
            # write a function that given a pandas dataframe df 
            # applies the clean_text function to a dataframe column 
            # called text and adds a new column called cleaned_text
            
            def clean_df(self):
                from datasets.datasets import clean_df
                self.df['cleaned_text'] = clean_df(self.df[self.text_column])
                return self.df
        
            # write a function that removes stopwords from text
            def remove_stopwords(self, text):
                # get the list of stopwords from the spacy library
                stop_words = STOP_WORDS
                # tokenize the text
                tokenized_text = self.nlp(text)
                # remove stopwords
                text = [word for word in tokenized_text if word not in stop_words]
                return text
        
            # write a function that given a pandas dataframe df 
            # applies the remove_stopwords function to a dataframe column 
            # called cleaned_text and adds a new column called cleaned_text_no_stopwords
            
            def remove_stopwords_df(self):
                self.df['cleaned_text_no_stopwords'] = self.df['cleaned_text'].apply(self.remove_stopwords)
                return self.df
        
            # write a function that lemmatizes words in text
            def lemmatize_words(self, text):
                # create a lemmatizer object from the spacy library
                lemmatizer = self.nlp(text)
                # lemmatize the words in the text
                text = [word.lemma_ for word in lemmatizer]
                return text
        
            # write a function that given a pandas dataframe df 
            # applies the lemmatize_words function to a dataframe column 
            # called cleaned_text and adds a new column called cleaned_text_lemmatized
            
            def lemmatize_words_df(self):
                self.df['cleaned_text_lemmatized'] = self.df['cleaned_text'].apply(self.lemmatize_words)
                return self.df
        
            # write a function that tokenizes text
            def tokenize_text(self, text):
                # tokenize the text
                tokenized_text = self.nlp(text)
                return tokenized_text
        
            # write a function that given a pandas dataframe df 
            # applies the tokenize_text function to a dataframe column
            # called cleaned_text and adds a new column called tokenized_text

            def tokenize_text_df(self):
                self.df['tokenized_text'] = self.df['cleaned_text'].apply(self.tokenize_text)
                return self.df
            
            # write a function that extracts named entities from text
            def extract_named_entities(self, text):
                # tokenize the text
                tokenized_text = self.nlp(text)
                # extract named entities
                named_entities = tokenized_text.ents
                return named_entities
            
            # write a function that given a pandas dataframe df
            # applies the extract_named_entities function to a dataframe column
            # called cleaned_text and adds a new column called named_entities

            def extract_named_entities_df(self):
                self.df['named_entities'] = self.df['cleaned_text'].apply(self.extract_named_entities)
                return self.df
            
            # write a function that extracts noun chunks from text
            def extract_noun_chunks(self, text):
                # tokenize the text
                tokenized_text = self.nlp(text)
                # extract noun chunks
                noun_chunks = tokenized_text.noun_chunks
                return noun_chunks
            
            # write a function that given a pandas dataframe df
            # applies the extract_noun_chunks function to a dataframe column
            # called cleaned_text and adds a new column called noun_chunks

            def extract_noun_chunks_df(self):
                self.df['noun_chunks'] = self.df['cleaned_text'].apply(self.extract_noun_chunks)
                return self.df

            # write a function that extracts parts of speech from text
            def extract_parts_of_speech(self, text):
                # tokenize the text
                tokenized_text = self.nlp(text)
                # extract parts of speech
                parts_of_speech = tokenized_text.pos_
                return parts_of_speech

            # write a function that given a pandas dataframe df
            # applies the extract_parts_of_speech function to a dataframe column
            # called cleaned_text and adds a new column called parts_of_speech

            def extract_parts_of_speech_df(self):
                self.df['parts_of_speech'] = self.df['cleaned_text'].apply(self.extract_parts_of_speech)
                return self.df

            # write a function that extracts dependencies from text
            def extract_dependencies(self, text):
                # tokenize the text
                tokenized_text = self.nlp(text)
                # extract dependencies
                dependencies = tokenized_text.dep_
                return dependencies

            # write a function that given a pandas dataframe df
            # applies the extract_dependencies function to a dataframe column
            # called cleaned_text and adds a new column called dependencies

            def extract_dependencies_df(self):
                self.df['dependencies'] = self.df['cleaned_text'].apply(self.extract_dependencies)
                return self.df

            # write a function that extracts sentences from text
            def extract_sentences(self, text):
                # tokenize the text
                tokenized_text = self.nlp(text)
                # extract sentences
                sentences = list(tokenized_text.sents)
                return sentences

            # write a function that given a pandas dataframe df
            # applies the extract_sentences function to a dataframe column
            # called cleaned_text and adds a new column called sentences

            def extract_sentences_df(self):
                self.df['sentences'] = self.df['cleaned_text'].apply(self.extract_sentences)
                return self.df

            # write a function that extracts tokens from text
            def extract_tokens(self, text):
                # tokenize the text
                tokenized_text = self.nlp(text)
                # extract tokens
                tokens = list(tokenized_text)
                return tokens

            # write a function that given a pandas dataframe df
            # applies the extract_tokens function to a dataframe column
            # called cleaned_text and adds a new column called tokens

            def extract_tokens_df(self):
                self.df['tokens'] = self.df['cleaned_text'].apply(self.extract_tokens)
                return self.df

            # write a function that extracts lemmas from text
            def extract_lemmas(self, text):
                # tokenize the text
                tokenized_text = self.nlp(text)
                # extract lemmas
                lemmas = [token.lemma_ for token in tokenized_text]
                return lemmas

            # write a function that given a pandas dataframe df
            # applies the extract_lemmas function to a dataframe column
            # called cleaned_text and adds a new column called lemmas

            def extract_lemmas_df(self):
                self.df['lemmas'] = self.df['cleaned_text'].apply(self.extract_lemmas)
                return self.df
            
            # write a function that extracts word vectors from text
            def extract_word_vectors(self, text):
                # tokenize the text
                tokenized_text = self.nlp(text)
                # extract word vectors
                word_vectors = [token.vector for token in tokenized_text]
                return word_vectors
            
            # write a function that given a pandas dataframe df
            # applies the extract_word_vectors function to a dataframe column
            # called cleaned_text and adds a new column called word_vectors

            def extract_word_vectors_df(self):
                self.df['word_vectors'] = self.df['cleaned_text'].apply(self.extract_word_vectors)
                return self.df

            # write a function that extracts similarity scores from text
            def extract_similarity_scores(self, text):
                # tokenize the text
                tokenized_text = self.nlp(text)
                # extract similarity scores
                similarity_scores = [token.similarity(tokenized_text[0]) for token in tokenized_text]
                return similarity_scores

            # write a function that given a pandas dataframe df
            # applies the extract_similarity_scores function to a dataframe column
            # called cleaned_text and adds a new column called similarity_scores

            def extract_similarity_scores_df(self):
                self.df['similarity_scores'] = self.df['cleaned_text'].apply(self.extract_similarity_scores)
                return self.df

            # write a function that extracts sentiment scores from text
            def extract_sentiment_scores(self, text):
                # tokenize the text
                tokenized_text = self.nlp(text)
                # extract sentiment scores
                sentiment_scores = tokenized_text.sentiment
                return sentiment_scores

            # write a function that given a pandas dataframe df
            # applies the extract_sentiment_scores function to a dataframe column
            # called cleaned_text and adds a new column called sentiment_scores

            def extract_sentiment_scores_df(self):
                self.df['sentiment_scores'] = self.df['cleaned_text'].apply(self.extract_sentiment_scores)
                return self.df
            
               


