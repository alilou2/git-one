import re
import spacy
import emoji
import numpy as np
#nltk.download('stopwords')
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize , RegexpTokenizer
from nltk.stem import SnowballStemmer
import qalsadi.lemmatizer
import tensorflow as tf
from collections import defaultdict
from tensorflow import keras
from transformers import  AutoTokenizer, AutoModelForSequenceClassification , pipeline
from transformers import BertTokenizer,TFBertModel


# Load stopwords for each language
with open('Algerian-Arabic-stop-words.txt', 'r', encoding='utf-8') as f:
    stop_words_ar_dz = set([line.strip() for line in f])
stop_words_en = set(stopwords.words('english'))
stop_words_fr = set(stopwords.words('french'))
# Create a custom tokenizer for Arabic words
tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')


with open('regroupement.txt', 'r',encoding='utf-8') as f:
    lines = f.readlines()
    dictionary = {}
    for line in lines:
        key, values = line.split(':')
        values = values.strip().split(',')
        for value in values:
            dictionary[value.strip()] = key.strip()

# Define a function to remove stop words from a text
def strip_emoji(text):
    return emoji.replace_emoji(text,replace="")


def remove_stopwords(text):
    words = tokenizer.tokenize(text)
    words_filtered = []
    for word in words:
        if word not in stop_words_ar_dz and word not in stop_words_en and word not in stop_words_fr:
            words_filtered.append(word)
    return ' '.join(words_filtered)


# Define a function to perform the text cleaning operations
def clean_text(text):
    text = text.replace("_", " ").replace("⁰", "")
    text = re.sub(r'[\r\n]+', ' ', text).lower()  
    text = re.sub(r"(?:\@|https?\://)\S+", "", text) #URLs
    #text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation marks 
    text = re.sub(r'\b\d+\b', '', text) # Remove only standalone numbers
    text = re.sub(r'[@#&$]\w+', '', text)  # Remove special characters
    text = re.sub(r'(\w)\1+', r'\1', text) # remove duplicated letters
    return text

def remove_mult_spaces(text):
    return re.sub("\s\s+" , " ", text)


def abreviation(text):
    words = text.split()
    new_words = [dictionary.get(word, word) for word in words]
    return ' '.join(new_words)

#load the model and tokenizer 
model_lang = AutoModelForSequenceClassification.from_pretrained('saved_model_langdetec')
tokenizer_lang = AutoTokenizer.from_pretrained('saved_model_langdetec')

classifier = pipeline("text-classification", model=model_lang, tokenizer=tokenizer_lang)

def lang_det(texte):
    texte=texte[:514]
    result = classifier(texte)
    
    if result[0]["score"] >= 0.2:
        langage = result[0]["label"]
    else:
        langage = 'dz'
    if langage not in ['en']:#['en','fr']
        langage = 'dz'
    
    
    return langage

# Define the mapping dictionary
mapping = {
    'aa':'ا','ai':'ي','ae':'ع','au': 'و','tion':'صيو', 'gh':'غ', 'kh':'خ','sh':'ش','dj':'ج','zz': 'ز','dh':'ض',
    'ch':'ش','ga': 'ڨ','go': 'ڨ','g': 'ج', 'h': 'ه', 'i': 'ي', 'j': 'ج', 'k': 'ك', 'l': 'ل','e': '', 'f': 'ف',
    'm': 'م', 'n': 'ن', 'ou': 'و','a': '','c':'ك', 'o': '', 'p': 'ب', 'q': 'ك', 'r': 'ر','d': 'د',
    's': 'س', 't': 'ت', 'u': 'ي', 'v': 'ف', 'w': 'و', 'x': 'كس','b': 'ب','6':'ط','2':'ا','8':'',
    'y': 'ي', 'z': 'ز','9':'ق','7':'ح','é':'','3':'ع','5':'خ','0':'','1':'','4':'',
}

# Define the translation function
def arabizi_to_arabic(text):

    pattern = '[ĩèāêôãûÎŠâɱíóøçïčýù]'
    
    text = re.sub(pattern, '', text)
    # Replace each Arabizi letter with its corresponding Arabic alphabet dialect letter
    for letter, value in mapping.items():
        
        text = re.sub(letter, value, text)
    return text

def stem_words(text , language):
    stemmer = None
    words = tokenizer.tokenize(text)
    if language == 'dz':
        text=arabizi_to_arabic(text)
        stemmed_text = text
    elif language == 'fr':
        stemmer = SnowballStemmer('french')
        stemmed_words = [stemmer.stem(word) for word in words]
        stemmed_text = ' '.join(stemmed_words)
    elif language == 'en':
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in words]
        stemmed_text = ' '.join(stemmed_words)
    else:
        return text # No stemmer available for this language
    
    return stemmed_text

nlp_en = spacy.load('en_core_web_sm')
nlp_fr = spacy.load('fr_core_news_sm')
def lemmatize(text , lang):
    # Tokenize the text
    words = tokenizer.tokenize(text)
    # Define lemmatizer for each language
    lemma_dz = qalsadi.lemmatizer.Lemmatizer()
    # Lemmatize each word in the text
    lemmatized_words = []
    if lang =='en':
        doc = nlp_en(text)
        for token in doc:
            lemma = token.lemma_
            lemmatized_words.append(lemma)
    elif lang =='fr':
        doc = nlp_fr(text)
        for token in doc:
            lemma = token.lemma_
            lemmatized_words.append(lemma)
    elif lang =='dz':
        for word in words:
            lemma = lemma_dz.lemmatize(word)
            lemmatized_words.append(lemma)
    
    # Join the lemmatized words to form the lemmatized text
    lemmatized_text = ' '.join(lemmatized_words)
    
    return lemmatized_text

def preprocess1(text):
    text = strip_emoji(text)
    text = clean_text(text)
    text = remove_mult_spaces(text)
    text = abreviation(text)
    return text

def preprocess2(text):
    langue = lang_det(text)
    text = stem_words(text , langue)
    text = remove_stopwords(text)
    text = lemmatize(text , langue)
    return text, langue

def prepare_data(input_text, tokenizer):
    token = tokenizer.encode_plus(
        input_text,
        max_length=256, 
        truncation=True, 
        padding='max_length', 
        add_special_tokens=True,
        return_tensors='tf'
    )
    return {
        'input_ids': tf.cast(token.input_ids, tf.float64),
        'attention_mask': tf.cast(token.attention_mask, tf.float64)
    }
def make_predictionprob(model, processed_data, classes=['Neutral', 'Hateful', 'Offensive']):
    
    probs = model.predict(processed_data)[0]
    class_probabilities = {cls: prob for cls, prob in zip(classes, probs)}
    return class_probabilities


def make_prediction(model, processed_data, classes=['Neutral', 'hateful','affensive']):
    probs = model.predict(processed_data)[0]
    return classes[np.argmax(probs)]


def predictComments(text, models,tokenizerbrt):
 try:
    input_text = preprocess1(text)
    input_text, langue = preprocess2(input_text)
    processed_data = prepare_data(input_text, tokenizerbrt)
    response = {}
    if langue == 'dz':
        result = make_predictionprob(models['dz'], processed_data=processed_data)
        print("Predicted probabilities:")
        for cls, prob in result.items():
     
            response[cls] = np.float64(prob)
    elif langue == 'en':
        result = make_predictionprob(models['en'], processed_data=processed_data)
        print("Predicted probabilities:")
        for cls, prob in result.items():
         
            response[cls] = np.float64(prob)
    print(response)        
    return response
 except Exception as e:
     print(f"error:{e}")

def predictlistComments(text, models,tokenizerbrt):
 try:
    input_text = preprocess1(text)
    input_text, langue = preprocess2(input_text)
    processed_data = prepare_data(input_text, tokenizerbrt)
    response = {}
    if langue == 'dz':
        result = make_prediction(models['dz'], processed_data=processed_data)
        return result
    elif langue == 'en':
        result = make_prediction(models['en'], processed_data=processed_data)
        return result
 except Exception as e:
     print(f"error:{e}")

