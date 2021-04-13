# coding=utf-8
import pandas as pd
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.corpus import stopwords
nltk.download('brown')
from nltk.corpus import brown
import collections
import operator
nltk.download('stopwords')
import secrets

import torch
from transformers import BertModel, BertTokenizer

#------------------------------------------------------------------------------------#
#
#   Embed text with BERT
#
#------------------------------------------------------------------------------------#

def compute_bert_embeddings(text, device=torch.device('cpu')):
    pretrained_weights = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
    model = BertModel.from_pretrained(pretrained_weights)

    return compute_transformer_embeddings(model, tokenizer, text, device)

def compute_transformer_embeddings(model, tokenizer, data, device):
    model = model.to(device)

    with torch.no_grad():
        data = [text.lower() for text in data]
        tokenized_data = tokenizer.batch_encode_plus(data, pad_to_max_length=True)
        tensor_data = torch.tensor(tokenized_data['input_ids'])
        tensor_data = tensor_data.to(device)

        predictions = model(tensor_data).last_hidden_state[:,0]
        embs = predictions

    return (embs,)

def transformer_embed_data(model, tokenizer, data):
    text_embeddings = None

#------------------------------------------------------------------------------------#
#
#   Common words
#
#------------------------------------------------------------------------------------#

def build_common_lemmas (refresh = False):

    if (refresh):

        punctuation = ['!','"','#','$','%','&',"'",'(',')','*','+',',','-','.','/',':',';','<','=','>','?','@','[','\\',']','^','_','`','{','|','}','~','``',"''",'--']

        nouns = {word for word, pos in brown.tagged_words() if pos.startswith('NN')}

        lower_words = [x.lower() for x in nouns]

        pun_stop = punctuation + stopwords.words('english')

        filter_words1 = [x for x in lower_words if x not in pun_stop]

        filter_words = list(filter(lambda x: x.isalpha() and len(x) > 1, filter_words1)) # remove numbers and single letter words

        words_count = dict(collections.Counter(filter_words))
        sorted_words = sorted(words_count.items(), key = operator.itemgetter(1), reverse = True)

        # first 5000 most commonly-occuring words
        V = [x[0] for x in sorted_words[:5000]]
        #C = V[:1000]

        filter_lemmas = [x for x in wordnet.words() if x in V]

        df = pd.DataFrame(filter_lemmas,columns=["Lemma"])

        df.to_pickle("common_words")
    else:
        df = pd.read_pickle("common_words")

    return df

#------------------------------------------------------------------------------------#
#
#   Build training data
#
#------------------------------------------------------------------------------------#

def build_training_data(df_common_lemmas, refresh=False):

    print (df_common_lemmas)

    df_training_data = pd.DataFrame(columns=["Lemma", "Positive","Negative"])

    if (refresh):

        for index, row in df_common_lemmas.iterrows():
            lemma = row["Lemma"]
            synonyms = wordnet.synsets(lemma)
            raw_hypernyms = []
            for syn in synonyms:
                for hyper in syn.hypernyms():
                   raw_hypernyms = raw_hypernyms + hyper.lemma_names()
            hypernyms = [x for x in raw_hypernyms if x in df_common_lemmas["Lemma"].values]
            if (hypernyms):
                pos = secrets.choice(hypernyms)
                neg = secrets.choice(df_common_lemmas["Lemma"].values)
                i = 1
                while ((neg in hypernyms) and (i < 10)):
                    neg = secrets.choice(df_common_lemmas["Lemma"].values)
                    i = i + 1
                if (i < 10):
                    new_row = {'Lemma': lemma, 'Positive': pos, 'Negative': neg}
                    df_training_data = df_training_data.append(new_row, ignore_index=True)
        df_training_data.to_pickle("training_data")
    else:
        df_training_data = pd.read_pickle("training_data")
    return df_training_data

#------------------------------------------------------------------------------------#
#
#   Main function
#
#------------------------------------------------------------------------------------#

df_common_lemmas = build_common_lemmas(refresh=False)
df_training_data = build_training_data(df_common_lemmas,refresh=False)
print(df_training_data)
data = df_training_data['Lemma'][:3]

print(data)

print (compute_bert_embeddings(data))