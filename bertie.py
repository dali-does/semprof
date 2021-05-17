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
#    tokenizer.add_special_tokens({'pad_token':'[PAD]'})

    model = model.from_pretrained(pretrained_weights)

    model.resize_token_embeddings(len(tokenizer))

    return compute_transformer_embeddings(model, tokenizer, text, device)

def compute_bert_embeddings(text, device=torch.device('cpu')):
    pretrained_weights = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
    model = BertModel.from_pretrained(pretrained_weights)

    return compute_transformer_embeddings(model, tokenizer, text, device)

def compute_transformer_embeddings(model, tokenizer, data, device):
    model = model.to(device)

    with torch.no_grad():
        data = [text.lower() for text in data]
        tokenized_data = tokenizer.batch_encode_plus(data, pad_to_max_length=True)#, add_special_tokens=True)
        tensor_data = torch.tensor(tokenized_data['input_ids'])
        tensor_data = tensor_data.to(device)

        predictions = model(tensor_data).last_hidden_state[:,0]
        embs = predictions

    return embs


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
        V = [x[0] for x in sorted_words[:8000]]
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
                nyms += syn.lemma_names()
                #syn_alts = get_altnyms(syn, nym)
                #for syn_alt in syn_alts:
                #   nyms += syn_alt.lemma_names()
            nyms = [x for x in nyms if x in df_common_lemmas["Lemma"].values]
            if (nyms):
                pos = secrets.choice(nyms)
                while pos is lemma:
                    pos = secrets.choice(nyms)
                neg = secrets.choice(df_common_lemmas["Lemma"].values)
#                import ipdb; ipdb.set_trace()
                i = 1
                while ((neg in nyms) and (i < 10)):
                    neg = secrets.choice(df_common_lemmas["Lemma"].values)
                    i = i + 1
                if (i < 10):
                    new_row = {'Lemma': lemma, 'Positive': pos, 'Negative': neg}
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
#                import ipdb; ipdb.set_trace()
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
#        __import__('ipdb').set_trace()
        for i, data in enumerate(dataloader):
            inputs, labels = data

            outputs = probe(inputs)
            _, preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    probe.train()
    return 100*correct/total

def sample_probe(probe, dataloader, original_data, num_samples=10):
    dataset = dataloader.dataset
    dataset_size = len(dataset)

    # Get a random sample
    random_index = np.random.randint(0,dataset_size, num_samples)

    sampler = torch.utils.data.SubsetRandomSampler(random_index)
    sample_loader = torch.utils.data.DataLoader(dataset=dataset, shuffle=False, batch_size=1, sampler=sampler)

    total = 0
    correct = 0
    probe.eval()
    with torch.no_grad():
#        __import__('ipdb').set_trace()
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

            print("Predicted "+pred_str + "(" +pred_item +") was " + label_str + "("+label_item+") for lemma "+orig_item["Lemma"])
            #print(labels)
            #print(preds)
            #print(original_data.iloc[random_index[i]])
    print("Accuracy: ", (100*correct/total))
    probe.train()

def kfold_train_eval(embedder, df_data, k_folds=5):

    kfold = KFold(n_splits=k_folds, shuffle=False)
    dataset, embedding_dim = df_to_dataset(embedder, df_data)
    print("Embedding dimension: ",embedding_dim/2)

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
        trained_probe = train_probe(probe, trainloader, num_epochs=20)

        accs.append(eval_probe(trained_probe, testloader))
        #sample_probe(trained_probe, dataloader, df_test_data, num_samples=10)
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)

    print("Accuracy: ", mean_acc, std_acc)


def df_to_dataset(embedder, df_data, seed=1971):
    embeddings = embedder(df_data['Lemma'])

#    __import__('ipdb').set_trace()
    pos_or_neg = df_data[['Positive','Negative']].apply(lambda row : row.sample(),axis=1)
    y = torch.tensor(pos_or_neg['Positive'].notna().astype(int).values, dtype=torch.long)
    pos_ratio = torch.sum(y)/len(y)
    print("Ratio of positive samples in dataset: ",pos_ratio)

    pos_or_neg = pos_or_neg.bfill(axis=1).iloc[:,0]
    pos_or_neg = embedder(pos_or_neg)

    joined_embeddings = torch.hstack((embeddings, pos_or_neg))

    return build_dataset(joined_embeddings, y), joined_embeddings[0].shape[-1]

df_common_lemmas = build_common_lemmas(refresh=False)


#df_training_data = df_data.sample(frac=0.8, random_state=1972)
#df_test_data = df_data.drop(df_training_data.index).sample(frac=1.0)

albert = lambda text: compute_huggingface_embeddings(text, tokenizer=AlbertTokenizer, model=AlbertModel, pretrained_weights='albert-base-v2')
roberta = lambda text: compute_huggingface_embeddings(text, tokenizer=RobertaTokenizer, model=RobertaModel, pretrained_weights='roberta-base')
#gpt = lambda text: compute_huggingface_embeddings(text, tokenizer=GPT2Tokenizer, model=GPT2Model, pretrained_weights='gpt2')
bert = lambda text: compute_huggingface_embeddings(text)

k_folds = 5
print("Albert, Roberta, BERT")
for nym in ['mero', 'hyper', 'hypo']:
    df_data = build_data_dynamic(df_common_lemmas, nym, refresh=True)
    print("Evaluating ",nym, " on ",len(df_data))
    for embedder in [albert, roberta, bert]:
    #for embedder in [gpt]:
        kfold_train_eval(embedder, df_data)

#embeddings = embedder(df_training_data['Lemma'])
#
#pos_or_neg = df_training_data[['Positive','Negative']].apply(lambda row : row.sample(),axis=1)
#y = torch.tensor(pos_or_neg['Positive'].notna().astype(int).values, dtype=torch.long)
#
#pos_or_neg = pos_or_neg.bfill(axis=1).iloc[:,0]
#pos_or_neg = embedder(pos_or_neg)
#
#num_classes = 2
#embedding_dim = 2*embeddings.shape[1]
#
#joined_embeddings = torch.hstack((embeddings, pos_or_neg))
##__import__('ipdb').set_trace()
#
#train_dataloader = build_dataset(joined_embeddings, y, batch_size=32)
#probe = LinearProbingModel(embedding_dim, num_classes)
#trained_probe = train_probe(probe, dataloader, num_epochs=20)
#
## 0 = neg, 1 = pos
#pos_or_neg = df_test_data[['Positive','Negative']].apply(lambda row : row.sample(),axis=1)
#y = torch.tensor(pos_or_neg['Positive'].notna().astype(int).values, dtype=torch.long)
#pos_or_neg = pos_or_neg.bfill(axis=1).iloc[:,0]
#pos_or_neg = embedder(pos_or_neg)
#
#embeddings = embedder(df_test_data['Lemma'])
#joined_embeddings = torch.hstack((embeddings, pos_or_neg))
#tes_dataloader = build_dataset(joined_embeddings, y, batch_size=32)
#
#eval_probe(trained_probe, dataloader)
#pos_ratio = torch.sum(y)/len(y)
#print("Ratio of positive samples in test: ",pos_ratio)
#sample_probe(trained_probe, dataloader, df_test_data, num_samples=500)
