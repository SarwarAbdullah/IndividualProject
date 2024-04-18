import pandas as pd
import nltk
import os
from difflib import SequenceMatcher
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.corpus import wordnet

positive_nltk_entities_books = None
positive_nltk_entities_movie_reviews = None
positive_nltk_entities_product_reviews = None
positive_nltk_entities_tweets = None
positive_nltk_entities_miscellaneous = None

def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return synonyms


def calculate_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def reorder_df(df, named_entities):
    df['similarity'] = df['named_entities'].apply(lambda x: calculate_similarity(x, named_entities))
    df = df.sort_values(by='similarity', ascending=False)
    df = df.drop(columns=['similarity'])
    return df

def match_keywords(df, keyword_list, working_column='cleaned_text_no_stopwords', query_sentiment = 'positive'):

    df[working_column] = df[working_column].apply(ast.literal_eval)

    try:
        print(df[working_column])
    except:
        print("\n\nWorking column: ", working_column, " not found in the dataframe\n\n")
        print("\n\ndataframe columns: ", df.columns, "\n\n")
    
    named_entities = get_named_entities_from_keyword_list(keyword_list)

    named_entities = ' '.join(named_entities)

    print(keyword_list)

    keyword_list = [word for phrase in keyword_list for word in phrase.split()] 

    # print the new keyword list
    print(keyword_list)

    # get synonyms

    synonyms = {word: get_synonyms(word) for word in keyword_list}

    keyword_list_with_synonyms = keyword_list.copy()

    for word in synonyms:
        keyword_list_with_synonyms.extend(synonyms[word])

    # remove duplicates
    keyword_list_with_synonyms = list(set(keyword_list))

    print(keyword_list_with_synonyms)

    def count_matches(words):
        return len(set(words) & set(keyword_list_with_synonyms))
    
    df['matches'] = df[working_column].apply(count_matches)

    df = df.sort_values(by='matches', ascending=False)

    matched_df = df[df['matches'] > 0]

    
    positive_df = matched_df[df['sentiment'] == 'positive']
    negative_df = matched_df[df['sentiment'] == 'negative']

    print(positive_df)

    
    for index, row in positive_df[positive_df['matches'] == True].iterrows():
        print(row[working_column])


    if positive_df[positive_df['matches'] == True].shape[0] == 0:
        print("\n\n================ matching words == False  ================\n\n")
        print("\n\n Unable to find any matching words in the df\n\n")
        print("\n\n")
        
        

    positive_df = positive_df.copy()
    negative_df = negative_df.copy()
    positive_df.loc[:, 'clean_text_no_stopwords_str'] = positive_df[working_column].apply(' '.join)
    negative_df.loc[:, 'clean_text_no_stopwords_str'] = negative_df[working_column].apply(' '.join)

    pcorpus = positive_df['clean_text_no_stopwords_str'].tolist() + [' '.join(keyword_list)]
    ncorpus = negative_df['clean_text_no_stopwords_str'].tolist() + [' '.join(keyword_list)]

    sorted_df = None
    if query_sentiment == 'positive':
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(pcorpus)

        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix).flatten()

        positive_df['cosine_similarity'] = cosine_similarities[:-1]

        print(cosine_similarities)
        print(cosine_similarities[0])

        positive_df_sorted = positive_df.sort_values(by='cosine_similarity', ascending=False)
        print("\n\n================ positive df sorted head ================\n\n")
        print(positive_df_sorted.head())
        print("\n\n==========================================\n\n ")


        if positive_df.shape[0] > 20:
            positive_df = positive_df.head(20)          
            positive_df = positive_df.sort_values(by=['retweet_count', 'favorite_count', 'mention_count', 'quote_retweet_count', 'quote_mention_count'], ascending=False)
        sorted_df = positive_df
    else:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(negative_df[working_column])
        cosine_similarities = cosine_similarity(tfidf_matrix, vectorizer.transform(keyword_list))

        negative_df['similarity'] = cosine_similarities

        negative_df = negative_df.sort_values(by='similarity', ascending=False)

        if negative_df.shape[0] > 20:
            negative_df = negative_df.head(20)
            negative_df = negative_df.sort_values(by=['retweet_count', 'favorite_count', 'mention_count', 'quote_retweet_count', 'quote_mention_count'], ascending=False)
        sorted_df = negative_df

    return sorted_df

def get_named_entities_from_keyword_list( keyword_list):
    print(keyword_list)
    named_entities = []
    import spacy
    nlp = spacy.load('en_core_web_sm')
    for keyword in keyword_list:
        doc = nlp(keyword)
        for ent in doc.ents:
            named_entities.append(ent.text)
    print(named_entities)
    return named_entities

def get_named_entities_from_nltk( root_dir ):
    global positive_nltk_entities_books
    global positive_nltk_entities_movie_reviews
    global positive_nltk_entities_product_reviews
    global positive_nltk_entities_tweets
    global positive_nltk_entities_miscellaneous

    #books
    from nltk.corpus import gutenberg
    books = gutenberg.fileids()
    all_books = gutenberg.raw(books)
    tokens = nltk.word_tokenize(all_books)
    pos_tags = nltk.pos_tag(tokens)
    positive_nltk_entities_books = nltk.ne_chunk(pos_tags)


    # movie reviews
    from nltk.corpus import movie_reviews
    pos_reviews = movie_reviews.fileids(categories='pos')
    neg_reviews = movie_reviews.fileids(categories='neg')

    all_reviews = pos_reviews + neg_reviews

    sample_review = movie_reviews.raw(all_reviews[0])

    tokens = nltk.word_tokenize(sample_review)

    pos_tags = nltk.pos_tag(tokens)

    positive_nltk_entities_movie_reviews = nltk.ne_chunk(pos_tags)

    ## product reviews
    from nltk.corpus import product_reviews_2
    product_reviews = product_reviews_2.fileids()
    all_product_reviews = product_reviews_2.raw(product_reviews)
    tokens = nltk.word_tokenize(all_product_reviews)
    pos_tags = nltk.pos_tag(tokens)
    positive_nltk_entities_product_reviews = nltk.ne_chunk(pos_tags)

    # tweets
    tweets = pd.read_csv(os.path.join(root_dir, 'datasets', 'tweets.csv'))
    tokens = nltk.word_tokenize(tweets['tweet'])
    pos_tags = nltk.pos_tag(tokens)
    positive_nltk_entities_tweets = nltk.ne_chunk(pos_tags)

def get_recommendations_from_nltk_reviews( named_entities ):
    
    global positive_nltk_entities_books
    global positive_nltk_entities_movie_reviews
    global positive_nltk_entities_product_reviews
    global positive_nltk_entities_tweets
    global positive_nltk_entities_miscellaneous

    # match the named entities to the positive nltk entities
    books = [entity for entity in positive_nltk_entities_books if entity in named_entities]
    movie_reviews = [entity for entity in positive_nltk_entities_movie_reviews if entity in named_entities]
    product_reviews = [entity for entity in positive_nltk_entities_product_reviews if entity in named_entities]
    tweets = [entity for entity in positive_nltk_entities_tweets if entity in named_entities]

    recommendations = books + movie_reviews + product_reviews + tweets

    return recommendations




