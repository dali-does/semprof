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
from transformers import GPT2Tokenizer, GPT2Model
from transformers import RobertaTokenizer, RobertaModel
from transformers import AlbertTokenizer, AlbertModel
from probing_model import LinearProbingModel

import numpy as np
from sklearn.model_selection import KFold

import gensim.downloader

#------------------------------------------------------------------------------------#
#
#   Embed text with Gensim
#
#------------------------------------------------------------------------------------#

def compute_gensim_embeddings(text, pretrained_model='word2vec-google-news-300'):
    model = gensim.downloader.load(pretrained_model)

    embeddings = torch.zeros((len(text), 300))
    for i, word in enumerate(text):
        if word in model:
            embeddings[i] = torch.tensor(model[word])
    return embeddings


#------------------------------------------------------------------------------------#
#
#   Embed text with BERT
#
#------------------------------------------------------------------------------------#
def compute_huggingface_embeddings(text, tokenizer=BertTokenizer,
                                   model=BertModel,
                                   pretrained_weights='bert-base-uncased',
                                   device=torch.device('cpu'),):
    tokenizer = tokenizer.from_pretrained(pretrained_weights)
    should_add_tokens = pretrained_weights.startswith('gpt')
    if should_add_tokens:
        tokenizer.add_special_tokens({'pad_token':'[PAD]'})
    model = model.from_pretrained(pretrained_weights)
    model.resize_token_embeddings(len(tokenizer))

    return compute_transformer_embeddings(model, tokenizer, text, device, should_add_tokens)

def compute_transformer_embeddings(model, tokenizer, data, device, special_tokens=False):
    model = model.to(device)

    with torch.no_grad():
        data = [text.lower() for text in data]
        tokenized_data = tokenizer.batch_encode_plus(data, pad_to_max_length=True, add_special_tokens=special_tokens)
        tensor_data = torch.tensor(tokenized_data['input_ids'])
        tensor_data = tensor_data.to(device)

        # Tested with [:, 1] as well to check if last hidden state used
        # makes difference, which for single word it doesn't
        predictions = model(tensor_data).last_hidden_state[:,0]
        embs = predictions
    return embs


#------------------------------------------------------------------------------------#
#
#   Common words
#
#------------------------------------------------------------------------------------#

def build_common_lemmas (num_most_common=5000, refresh = False):

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
        V = [x[0] for x in sorted_words[:num_most_common]]

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

def get_altnyms(syn, nym):
    if nym == 'hyper':
        return syn.hypernyms()
    if nym == 'mero':
        return syn.part_meronyms()
    if nym == 'hypo':
        return syn.hyponyms()
    return syn.synonyms()


def build_data_dynamic(df_common_lemmas, nym='hyper', refresh=False):

    df_data = pd.DataFrame(columns=["Lemma", "Positive","Negative"])

    if (refresh):

        for index, row in df_common_lemmas.iterrows():
            lemma = row["Lemma"]
            synonyms = wordnet.synsets(lemma)
            nyms = []
            for syn in synonyms:
                if nym == 'syn':
                    nyms += syn.lemma_names()
                else:
                    syn_alts = get_altnyms(syn, nym)
                    for syn_alt in syn_alts:
                       nyms += syn_alt.lemma_names()
            nyms = [x for x in nyms if x in df_common_lemmas["Lemma"].values and x != lemma]
            if (nyms):
                pos = secrets.choice(nyms)
                while (pos == lemma):
                    pos = secrets.choice(nyms)
                neg = secrets.choice(df_common_lemmas["Lemma"].values)
                i = 1
                while ((neg in nyms) or (neg == lemma)) and (i < 10):
                    neg = secrets.choice(df_common_lemmas["Lemma"].values)
                    i = i + 1
                if (i < 10):
                    new_row = {'Lemma': pos, 'Positive': lemma, 'Negative': neg}
                    df_data = df_data.append(new_row, ignore_index=True)
        df_data.to_pickle("training_data")
    else:
        df_data = pd.read_pickle("training_data")
    return df_data


def build_data(df_common_lemmas, refresh=False):

    df_data = pd.DataFrame(columns=["Lemma", "Positive","Negative"])

    if (refresh):

        for index, row in df_common_lemmas.iterrows():
            lemma = row["Lemma"]
            synonyms = wordnet.synsets(lemma)
            raw_hypernyms = []
            for syn in synonyms:
                for hyper in syn.hyponyms():
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
                    df_data = df_data.append(new_row, ignore_index=True)
        df_data.to_pickle("training_data")
    else:
        df_data = pd.read_pickle("training_data")
    return df_data

#------------------------------------------------------------------------------------#
#
#   Main function
#
#------------------------------------------------------------------------------------#

def build_dataset(x, y, batch_size=32):
    dataset = torch.utils.data.TensorDataset(x, y)
    return dataset
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    #return dataloader

def train_probe(probe, dataloader, num_epochs=5, print_loss=False):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.001)

    running_loss = 0.0
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            inputs, labels = data
            optimizer.zero_grad()
    
            outputs = probe(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            if print_loss and i % 100 == 0:
                print(epoch + 1, i + 1, running_loss / 100)
                running_loss = 0.0
    #print('Finished training')
    return probe

def eval_probe(probe, dataloader):
    total = 0
    correct = 0
    probe.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, labels = data

            outputs = probe(inputs)
            _, preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    probe.train()
    return 100*correct/total

def sample_probe(probe, dataset, original_data, num_samples=10):
    dataset_size = len(dataset)

    # Get a random sample
    random_index = np.random.randint(0,dataset_size, num_samples)

    sampler = torch.utils.data.SubsetRandomSampler(random_index)
    sample_loader = torch.utils.data.DataLoader(dataset=dataset, shuffle=False, batch_size=1, sampler=sampler)

    total = 0
    correct = 0
    probe.eval()
    with torch.no_grad():
        for i, data in enumerate(sample_loader):
            inputs, labels = data

            outputs = probe(inputs)
            _, preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            orig_item = original_data.iloc[random_index[i]]

            label_str = "negative" if labels[0] == 0 else "positive"
            pred_str = "negative" if preds[0] == 0 else "positive"

            label_item = orig_item["Negative"] if labels[0] == 0 else orig_item["Positive"]
            pred_item = orig_item["Negative"] if preds[0] == 0 else orig_item["Positive"]

            print("Triple: ", (orig_item["Lemma"], orig_item["Positive"], orig_item["Negative"]), "label: ", label_item, ", prediction was ", pred_item)
    print("Accuracy: ", (100*correct/total))
    probe.train()


def kfold_train_eval(embedder, df_data, pos_or_neg, y, k_folds=5):

    kfold = KFold(n_splits=k_folds, shuffle=False)
    dataset, embedding_dim = df_to_dataset(embedder, df_data, pos_or_neg, y)
    #print("Embedding dimension: ",embedding_dim/2)

    accs = []
    num_classes = 2

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        train_subset = torch.utils.data.Subset(dataset, train_ids)
        test_subset = torch.utils.data.Subset(dataset, test_ids)

        trainloader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=32)
        testloader = torch.utils.data.DataLoader(
            test_subset,
            batch_size=32)
        probe = LinearProbingModel(embedding_dim, num_classes)

        probe = LinearProbingModel(embedding_dim, num_classes)
        trained_probe = train_probe(probe, trainloader, num_epochs=10)

        accs.append(eval_probe(trained_probe, testloader))
    sample_probe(trained_probe, dataset, df_data, num_samples=10)
    return accs


def df_to_dataset(embedder, df_data, pos_or_neg, y):
    embeddings = embedder(df_data['Lemma'])
    pos_or_neg = embedder(pos_or_neg)
    joined_embeddings = torch.hstack((embeddings, pos_or_neg))
    return build_dataset(joined_embeddings, y), joined_embeddings[0].shape[-1]

def generate_y_from_df(df_data):
    pos_or_neg = df_data[['Positive','Negative']].apply(lambda row : row.sample(),axis=1)
    y = torch.tensor(pos_or_neg['Positive'].notna().astype(int).values, dtype=torch.long)
    control_y = torch.randint(0,2,(len(pos_or_neg),))
    pos_ratio = torch.sum(y)/len(y)
    print("Ratio of positive samples in dataset: ",pos_ratio)

    pos_or_neg = pos_or_neg.bfill(axis=1).iloc[:,0]

    return pos_or_neg, y, control_y

df_common_lemmas = build_common_lemmas(refresh=True)

word2vec = lambda text: compute_gensim_embeddings(text, pretrained_model='word2vec-google-news-300')
glove = lambda text: compute_gensim_embeddings(text, pretrained_model='glove-wiki-gigaword-300')

albert = lambda text: compute_huggingface_embeddings(text, tokenizer=AlbertTokenizer, model=AlbertModel, pretrained_weights='albert-base-v2')
roberta = lambda text: compute_huggingface_embeddings(text, tokenizer=RobertaTokenizer, model=RobertaModel, pretrained_weights='roberta-base')
gpt = lambda text: compute_huggingface_embeddings(text, tokenizer=GPT2Tokenizer, model=GPT2Model, pretrained_weights='gpt2')
bert = lambda text: compute_huggingface_embeddings(text)


use_control = False
k_folds = 5
print("word2vec, glove, Albert, Roberta, BERT, GPT2")
for nym in ['syn','mero', 'hyper', 'hypo']:
    np.random.seed(1974)
    df_data = build_data_dynamic(df_common_lemmas, nym, refresh=True)
    pos_or_neg, y, control_y = generate_y_from_df(df_data)
    print("Evaluating ",nym, " on ",len(df_data))
    for embedder in [word2vec, glove, albert, roberta, bert, gpt]:
        accuracies = []
        res = []
        if use_control:
            control_res = []
        for i in range(5):
            accs = kfold_train_eval(embedder, df_data, pos_or_neg, y)
            mean_acc = np.mean(accs)
            res.append(mean_acc)
            accuracies += accs

            if use_control: 
                control_mean_acc, control_std_acc = kfold_train_eval(embedder, df_data, pos_or_neg, control_y)
                control_res.append(control_mean_acc)
        #mean_acc = np.mean(res)
        #std_acc = np.std(res)
        #print("Accuracy: ", mean_acc, std_acc)
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        print("Accuracy: ", mean_acc, std_acc)
        
        if use_control:
            control_mean_acc = np.mean(control_res)
            control_std_acc = np.std(control_res)
            print("Accuracy (control): ", control_mean_acc, control_std_acc)
