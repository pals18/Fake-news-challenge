
import os
import re
import nltk
import numpy as np
from sklearn import feature_extraction
from tqdm import tqdm
import string
from collections import Counter
import math
import nltk
from gensim.matutils import kullback_leibler
from gensim.corpora import Dictionary
from gensim.models import ldamodel
import string
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer



### LOADING THE TRAIN AND TEST DATASETS

data='data'
df1=pd.read_csv(data+"//train_bodies.csv")
df2=pd.read_csv(data+"//train_stances.csv")
dfc2=pd.read_csv(data+"//competition_test_stances.csv")
dfc1=pd.read_csv(data+"//competition_test_bodies.csv")
tdf1=pd.merge(left=df2, right=df1, how='left', left_on='Body ID', right_on='Body ID')
tdf2=pd.merge(left=dfc2, right=dfc1, how='left', left_on='Body ID', right_on='Body ID')

### FUNCTIONS FOR REMOVING THE PUNCTUTAIONS AND TOKENIZING AND LEMMATIZING SENTENCES


_wnl = nltk.WordNetLemmatizer()
def normalize_word(w):
    return _wnl.lemmatize(w).lower()


def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in nltk.word_tokenize(s)]


def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric

    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()


def remove_stopwords(l):
    # Removes stopwords from a list of tokens
    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]


### FUNCTION FOR CALCULATING KL DIVERGENCE BETWEEN HEADLINE AND BODY


def kld (headlines, bodies):
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        total1=[]
        clean_headline = clean(headline)
        clean_body = clean(body)
        clean_headline1 = get_tokenized_lemmas(clean_headline)
        clean_body1 = get_tokenized_lemmas(clean_body)
        total=clean_body1 + clean_headline1
        total1.append(total)
#        print (total1)
    dictionary = Dictionary(total1)
    corpus = [dictionary.doc2bow(text) for text in total1]

    model = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=2, minimum_probability=1e-8)
            #model.show_topics()
    kld=[]

    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        clean_headline1 = get_tokenized_lemmas(clean_headline)
        clean_body1 = get_tokenized_lemmas(clean_body)
        bow_head = model.id2word.doc2bow(clean_headline1)
        bow_body = model.id2word.doc2bow(clean_body1)
        lda_bow_head = model[bow_head]
        lda_bow_body = model[bow_body]
        kld.append(kullback_leibler(lda_bow_head, lda_bow_body))
#        print(kld)
    return kld



### FUNCTION FOR CALCULATING THE JACCARD DISTANCE

list1=[]
list2=[]
list3=[]

def jacob(headlines,bodies) :
    list1=[]
    for i, (headline, body) in tqdm(enumerate(zip(headlines , bodies))):
        text1 = "".join([word for word in headline if word not in string.punctuation])
        text2 = "".join([word for word in body if word not in string.punctuation])
        a=set(text1.split())
        b=set(text2.split())
        c = a.intersection(b)
        lis=float(len(c)) / (len(a) + len(b) - len(c))
        list1.append(lis)


    return list1


### FUNCTION FOR CALCULATING THE COSINE SIMILARITY BETWEEN HEADLINE AND BODIES


# self created feature cosine similarity
def text_to_vector(text):
    WORD=re.compile(r"\w+")
    
    words= WORD.findall(text)
    return Counter(words)

def get_cosine(headlines, bodies):
    vec1 = text_to_vector(str(headlines))
    vec2 = text_to_vector(str(bodies))



    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
return 0.0
    else:
return float(numerator) / denominator


cosine1=[]
def cos_feat(headlines, bodies):
    cosine1=[]
    list2=[]
    list3=[]
    for i, (headline, body) in tqdm(enumerate(zip(headlines , bodies))):
        text1 = "".join([word for word in headline if word not in string.punctuation])
        text2 = "".join([word for word in body if word not in string.punctuation])
        list2.append(text1)
        list3.append(text2)
    for i, (headline, body) in tqdm(enumerate(zip(list2 , list3))):

cosine1.append(get_cosine(headline, body))
#        print(cosine1)
    return cosine1



### FUNCTION TO CALCULATE EUCLEDIAN DISTANCE


def text_to_vector(text):
    WORD=re.compile(r"\w+")
    
    words= WORD.findall(text)
    return Counter(words)
def get_euc(headlines, bodies):
    vec1 = text_to_vector(str(headlines))
    vec2 = text_to_vector(str(bodies))
    res=math.sqrt(sum((vec1.get(k, 0) - vec2.get(k, 0))**2 for k in set(vec1.keys()).union(set(vec2.keys()))))
    return res
    
        
euc=[]
def euc_feat(headlines, bodies):
    euc=[]
    list2=[]
    list3=[]
    for i, (headline, body) in tqdm(enumerate(zip(headlines , bodies))):
                text1 = "".join([word for word in headline if word not in string.punctuation])
                text2 = "".join([word for word in body if word not in string.punctuation])
                list2.append(text1)
                list3.append(text2)
    for i, (headline, body) in tqdm(enumerate(zip(list2 , list3))):

        euc.append(get_euc(headline, body))
#        print(cosine1)
    return euc


### FUNCTIONS FOR WORD OVERLAP, REFUTING FEATURES, POLARITY AND HAND FEATURES



def word_overlap_features(headlines, bodies):
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        clean_headline = get_tokenized_lemmas(clean_headline)
        clean_body = get_tokenized_lemmas(clean_body)
        features = [len(set(clean_headline).intersection(clean_body)) / float(len(set(clean_headline).union(clean_body)))]
        X.append(features)
    return X


def refuting_features(headlines, bodies):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        # 'refute',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_headline = get_tokenized_lemmas(clean_headline)
        features = [1 if word in clean_headline else 0 for word in _refuting_words]
        X.append(features)
    return X


def polarity_features(headlines, bodies):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]

    def calculate_polarity(text):
        tokens = get_tokenized_lemmas(text)
        return sum([t in _refuting_words for t in tokens]) % 2
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        features = []
        features.append(calculate_polarity(clean_headline))
        features.append(calculate_polarity(clean_body))
        X.append(features)
    return np.array(X)


def ngrams(input, n):
    input = input.split(' ')
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def chargrams(input, n):
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def append_chargrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in chargrams(" ".join(remove_stopwords(text_headline.split())), size)]
    grams_hits = 0
    grams_early_hits = 0
    grams_first_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
        if gram in text_body[:100]:
            grams_first_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    features.append(grams_first_hits)
    return features


def append_ngrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in ngrams(text_headline, size)]
    grams_hits = 0
    grams_early_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    return features


def hand_features(headlines, bodies):

    def binary_co_occurence(headline, body):
        # Count how many times a token in the title
        # appears in the body text.
        bin_count = 0
        bin_count_early = 0
        for headline_token in clean(headline).split(" "):
            if headline_token in clean(body):
                bin_count += 1
            if headline_token in clean(body)[:255]:
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def binary_co_occurence_stops(headline, body):
        # Count how many times a token in the title
        # appears in the body text. Stopwords in the title
        # are ignored.
        bin_count = 0
        bin_count_early = 0
        for headline_token in remove_stopwords(clean(headline).split(" ")):
            if headline_token in clean(body):
                bin_count += 1
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def count_grams(headline, body):
        # Count how many times an n-gram of the title
        # appears in the entire body, and intro paragraph

        clean_body = clean(body)
        clean_headline = clean(headline)
        features = []
        features = append_chargrams(features, clean_headline, clean_body, 2)
        features = append_chargrams(features, clean_headline, clean_body, 8)
        features = append_chargrams(features, clean_headline, clean_body, 4)
        features = append_chargrams(features, clean_headline, clean_body, 16)
        features = append_ngrams(features, clean_headline, clean_body, 2)
        features = append_ngrams(features, clean_headline, clean_body, 3)
        features = append_ngrams(features, clean_headline, clean_body, 4)
        features = append_ngrams(features, clean_headline, clean_body, 5)
        features = append_ngrams(features, clean_headline, clean_body, 6)
        return features
    
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
            X.append(binary_co_occurence(headline, body)
             + binary_co_occurence_stops(headline, body)
             + count_grams(headline, body))
             

    return X


### CALLING THE ABOVE FUNCTIONS


df=tdf1.append(tdf2, ignore_index=True)
X_overlap = word_overlap_features(df['Headline'], df['articleBody'])
X_refuting = refuting_features(df['Headline'], df['articleBody'])
X_polarity = polarity_features(df['Headline'], df['articleBody'])
X_hand = hand_features(df['Headline'], df['articleBody'])
X_cosine= cos_feat(df['Headline'], df['articleBody'])
X_kld= kld(df['Headline'], df['articleBody'])
X_jacob= jacob(df['Headline'], df['articleBody'])
X_euc=euc_feat(df['Headline'],df['articleBody'])



### COMBINING ALL THE FEATURES


X = np.column_stack((X_hand, X_polarity, X_refuting, X_overlap, X_cosine, X_kld, X_jacob,X_euc))

### CREATING A DATAFRAME OF X AND STORING IT IN A FILE PANDAS FILE


X=pd.DataFrame(X)

X.to_csv('features//features.csv',index=False)


### CREATING FEATURES USING TFIDF TRANSFORMER



df['combined']=df['Headline']+df['articleBody']


df['combined']=df['combined'].apply(clean)



df['combined']=df['combined'].apply(get_tokenized_lemmas)


df.to_csv("features//combinedf.csv",index=False)

count_vector      = CountVectorizer(ngram_range=(1,1),tokenizer=lambda doc: doc,max_features=25000, lowercase=False)
count_x_train     = count_vector.fit_transform(df['combined'])
tfidf_transformer = TfidfTransformer()
tfidf_x_train = tfidf_transformer.fit_transform(count_x_train)


with open('features//tfidf_25k.pkl', 'wb') as handle:
    pickle.dump(tfidf_x_train, handle, protocol=pickle.HIGHEST_PROTOCOL)



count_vector      = CountVectorizer(ngram_range=(1,1),tokenizer=lambda doc: doc, lowercase=False)
count_x_train     = count_vector.fit_transform(df['combined'])
tfidf_transformer = TfidfTransformer()
tfidf_x_train = tfidf_transformer.fit_transform(count_x_train)


with open('features//tfidf_all.pkl', 'wb') as handle:
    pickle.dump(tfidf_x_train, handle, protocol=pickle.HIGHEST_PROTOCOL)







