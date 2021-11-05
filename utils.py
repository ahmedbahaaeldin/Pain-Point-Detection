from transformers import RobertaTokenizer, RobertaModel,BertTokenizer,BertModel
import torch.nn as nn
import nlpaug.augmenter.word as naw
import re
import torch
import pandas as pd
import numpy as np
import math
from sklearn import metrics

def dataset_sampling(dataset,sample_size,random_state):
    if sample_size > len(dataset[dataset['Type of Pain'] == 'Product feature or quality']):
        d1 = dataset[dataset['Type of Pain'] == 'Product feature or quality']
    else:
        d1 = dataset[dataset['Type of Pain'] == 'Product feature or quality'].sample(sample_size,random_state=random_state)
        
    if sample_size > len(dataset[dataset['Type of Pain'] == 'Operational issues']):
        d2 = dataset[dataset['Type of Pain'] == 'Operational issues']
    else:
        d2 = dataset[dataset['Type of Pain'] == 'Operational issues'].sample(sample_size,random_state =random_state)   
    
    if sample_size > len(dataset[dataset['Type of Pain'] == 'Service quality or failure']):
        d3 = dataset[dataset['Type of Pain'] == 'Service quality or failure']
    else:
        d3 = dataset[dataset['Type of Pain'] == 'Service quality or failure'].sample(sample_size,random_state = random_state)   
    
    if sample_size > len(dataset[dataset['Type of Pain'] == "Company's image"]):
        d4 = dataset[dataset['Type of Pain'] == "Company's image"]
    else:
        d4 = dataset[dataset['Type of Pain'] == "Company's image"].sample(sample_size,random_state = random_state)  
     
    if sample_size > len(dataset[dataset['Type of Pain'] == 'Customer service']):
        d5 = dataset[dataset['Type of Pain'] == "Customer service"]
    else:
        d5 = dataset[dataset['Type of Pain'] == "Customer service"].sample(sample_size,random_state = random_state) 
        
    dataset = pd.concat((d1,d2,d3,d4,d5),axis=0)                       
    return dataset
         
                                 
def labels_preprocessing(dataset,label_column_name):
    labels = pd.get_dummies(dataset[label_column_name])
    dummies_names = labels.columns
    labels = labels.to_numpy()
    labels = torch.from_numpy(labels)
    return labels,dummies_names
                                 
                                 
def init_roberta_model(gpu_usage):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')
    if gpu_usage:
        model.to('cuda')
    return tokenizer,model
    
def init_bert_model(gpu_usage):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    if gpu_usage:
        model.to('cuda')
    return tokenizer,model

                                 
def preprocess_text(dataset,text_column_name):
    clean_dataset = []
    for sent in dataset[text_column_name]:
        new_sent = re.sub(r'\@[A-Za-z0-9\_]+','',sent)
        new_sent =  re.sub('[^0-9a-zA-Z\ \.\,]+','',new_sent)
        clean_dataset.append(new_sent)
    return clean_dataset

                                 
def transformer_model_dataset_preparation(dataset,model,tokenizer,max_length):
    model_output = []
    model.eval()
    with torch.no_grad():
        for sent in dataset:
            tokenized_sentence = tokenizer(sent,padding='max_length',max_length=max_length,truncation=True,return_tensors='pt')
            model_sent = model(**tokenized_sentence.to('cuda'))
            model_output.append(model_sent[0].detach().cpu().numpy())
            del model_sent
            del tokenized_sentence
    model_output = torch.from_numpy(np.array(model_output))
    return model_output


    
def train_test_split(dataset,labels,test_size=120):
    perm = torch.randperm(dataset.size()[0])
    full_dataset = dataset[perm]
    full_labels = labels[perm]
    test_dataset = full_dataset[-test_size:]
    test_labels = full_labels[-test_size:]
    train_dataset = full_dataset[:-test_size]
    train_labels = full_labels[:-test_size]
    return train_dataset,train_labels,test_dataset,test_labels

def classification_report(test_labels,test_out,dummies_names,output_dict):
    return metrics.classification_report(np.array(torch.argmax(test_labels.cpu(),dim=-1)),
np.array(torch.argmax(test_out.detach().cpu(),dim=-1)),labels=[0,1,2,3,4],target_names=dummies_names,output_dict=output_dict)

def training_loop(model,train_dataset,train_labels,test_dataset,test_labels,batch_size,epochs,learning_rate,dummies_names):
    n_batches = math.ceil(len(train_dataset) // batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    best_f1_score = 0
    for epoch in range(epochs):
        batch_losses = []
        for i in range(n_batches):
            model.train()
            optimizer.zero_grad()
            local_X, local_y = train_dataset[max(0,i * batch_size):min(len(train_dataset),(i+1) * batch_size)] , train_labels[max(0,i * batch_size):min(len(train_labels),(i+1) * batch_size)]
            output = model(local_X.float().cuda(),batch_size)
            loss = criterion(output,torch.max(local_y.long().cuda(),1)[1])
            batch_losses.append(loss.item())
            loss.backward()
            optimizer.step() 

        print('Loss value:'+str(sum(batch_losses)/len(batch_losses)))
        model.eval()
        with torch.no_grad():
            test_out = model(test_dataset.float().cuda(),len(test_dataset))
            test_out = nn.Softmax(dim=-1)(test_out)
            print('Test Results Epoch '+str(epoch)+' :')
            print(classification_report(test_labels,test_out,dummies_names,output_dict=False))
            print('------------------------------------------------------------------------------------------')
            test_results = classification_report(test_labels,test_out,dummies_names,output_dict=True)
            if test_results['weighted avg']['f1-score'] > best_f1_score:
                best_f1_score = test_results['weighted avg']['f1-score']
                class_1_f1 = test_results["Company's image"]['f1-score']
                class_2_f1 = test_results["Customer service"]['f1-score']
                class_3_f1 = test_results["Operational issues"]['f1-score']
                class_4_f1 = test_results["Product feature or quality"]['f1-score']
                class_5_f1 = test_results["Service quality or failure"]['f1-score']
                torch.save(model,'class_model.pth')
    print('Best Model Performance :')
    print('Weighted F1-Score:{}'.format(best_f1_score))
    print("Company's image F1-Score:{}".format(class_1_f1))
    print("Customer service F1-Score:{}".format(class_2_f1))
    print("Operational issues F1-Score:{}".format(class_3_f1))
    print("Product feature or quality F1-Score:{}".format(class_4_f1))
    print("Service quality or failure F1-Score:{}".format(class_5_f1))



def augment_text_substitute(text,augment_type):
    if augment_type == 'word_emb':
        aug = naw.WordEmbsAug(
            model_type='fasttext', model_path='wiki-news-300d-1M.vec',
            action="insert")
    elif augment_type == 'wordnet':
        aug = naw.SynonymAug(aug_src='wordnet')
    elif augment_type == 'contextual_emb':
        aug = naw.ContextualWordEmbsAug(
          model_path='distilbert-base-uncased', action="insert")
    aug_dataset = []
    for sent in text:
        augmented_text = aug.augment(sent)
        aug_dataset.append(augmented_text) 
    return aug_dataset 


def augment_text_random(text,action):
    if action == 'swap':
        aug = naw.RandomWordAug(action=action)
    elif action == 'crop':
        aug = naw.RandomWordAug(action=action)
    elif action == 'delete':
        aug = naw.RandomWordAug()
    aug_dataset = []
    for sent in text:
        augmented_text = aug.augment(sent)
        aug_dataset.append(augmented_text) 
    return aug_dataset 

