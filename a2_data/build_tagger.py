# -*- coding: utf-8 -*-
"""
Created on Mon Oct 09 12:47:21 2017

@author: lanyi
"""

#%%
from __future__ import division
import re
import math
import sys
import pickle
#%%
'''
                                                       Build tagger     
'''
#%%
def isNum(word):
    """
    To Judge whether a word is a number or not.
        
    INPUTS:
    - word: A single string token.
    
    OUTPUTS:
    - out: Boolean value. True if the word is a number, False if it is not.
    """
    pattern=re.compile(r'^[0-9]+\.?[0-9]*')
    result=pattern.match(word)
    if result:
        return True
    else:
        return False
#%%
def train_preprocess(train_view):
    """
    Preprocssing the training file and extract useful information.
        
    INPUTS:
    - train_view: A list of sentences from the training file.
    
    OUTPUTS:
    - tag_words: A dictionary. Key is the POS tag, Value is another dictionary which key is word and value is the word count.
    - bag_of_tag:A dictionary. Key is the POS tag, Value is the tag count.
    - bag_of_words:A dictionary. Key is the word, Value is another dictionary which key is POS tag and value is tag count.
    """
    tag_words={}
    bag_of_tag={}
    bag_of_words={}
    for sentence in train_view:
        for wordTag in sentence.split():
            tag=wordTag.split('/')[-1] 
            for word in wordTag.split('/')[:-1]:
                if word not in bag_of_words:
                    bag_of_words[word]={tag:1}
                else:
                    if tag not in bag_of_words[word]:
                        bag_of_words[word][tag]=1
                    else:
                        bag_of_words[word][tag]+=1
                if tag not in bag_of_tag:
                    bag_of_tag[tag]=1
                else:
                    bag_of_tag[tag]+=1
                if tag not in tag_words:
                    tag_words[tag]={word:1}
                else:
                    if word not in tag_words[tag]:
                        tag_words[tag][word]=1
                    else:
                        tag_words[tag][word]+=1
    return tag_words,bag_of_tag,bag_of_words
#%%
def tagTransitionCount(bag_of_tag, train_view):
    """
    Procssing the training file and get the tag transition count which will be used in Viterbi algorithm.
        
    INPUTS:
    - bag_of_tag: A dictionary of the POS tag.
    - train_view: A list of sentences from the training file.
    
    OUTPUTS:
    - tag_transition: A dictionary. Key is the previous POS tag, Value is another 
                    dictionary which key is next POS tag and value is the C(tag_n|tag_p).
                    Two more tag <s> and </s> are added.
    """
    tag_transition={}
    tag_transition['<s>']={}
    for x in bag_of_tag:
        tag_transition['<s>'].update({x:0})
    tag_transition['<s>'].update({'</s>':0})
    for tag in bag_of_tag:
        tag_transition[tag]={}
        for x in bag_of_tag:
            tag_transition[tag].update({x:0})
        tag_transition[tag].update({'</s>':0})
    
    for lines in train_view:
        pre_tag='<s>'
        for corpus in lines.split():
            tag=corpus.split('/')[-1]
            tag_transition[pre_tag][tag]+=1
            pre_tag=tag
        tag_transition[pre_tag]['</s>']+=1
    return tag_transition
#%%
def tagWordProb(tag_words):
    """
    Calculate the probability of a word given a tag P(word|tag).
        
    INPUTS:
    - tag_words: A dictionary. Key is the POS tag, Value is another dictionary which key is word and value is the word count.
    
    OUTPUTS:
    - tagWordP: A dictionary. Key is the POS tag, Value is another 
                dictionary which key is word and value is the P(word|tag).
    """
    tagWordP={}
    for tag in tag_words:
        total=sum(tag_words[tag].values())
        tagWordP[tag]={}
        for word in tag_words[tag]:
            tagWordP[tag].update({word: tag_words[tag][word]/total})
    return tagWordP
#%%
def tag_unknown(bag_of_words,bag_of_tag):
    """
    Calculate the probability of an unknown_word given a tag P(tag|unknownword). 
    The probability distribution of unknown word is similiar to word that occurs once in the training dataset.
        
    INPUTS:
    - bag_of_words:A dictionary. Key is the word, Value is another dictionary which key is POS tag and value is tag count.
    - bag_of_tag:A dictionary. Key is the POS tag, Value is the tag count.
    
    OUTPUTS:
    - probability: if a word is unknown, the probability distribution of its possible tag.
    """
    tag_unknown_count={}
    for tag in bag_of_tag:
        tag_unknown_count[tag]=0
    for word in bag_of_words:
        if(len(bag_of_words[word])==1 and bag_of_words[word].values()[0]==1):
            if bag_of_words[word].keys()[0] not in tag_unknown_count:
                tag_unknown_count[bag_of_words[word].keys()[0]]=1
            else:
                tag_unknown_count[bag_of_words[word].keys()[0]]+=1
    probability={}
    total=sum(tag_unknown_count.values())+len(tag_unknown_count)*0.01
    for tag in tag_unknown_count:
        probability[tag]=(tag_unknown_count[tag]+0.01)/total
    return probability
#%%
def tag_capital(bag_of_words,bag_of_tag):
    """
    Calculate the probability of P(isCapital|tag). 'c' is capital and 'nc' is not capital.
        
    INPUTS:
    - bag_of_words:A dictionary. Key is the word, Value is another dictionary which key is POS tag and value is tag count.
    - bag_of_tag:A dictionary. Key is the POS tag, Value is the tag count.
    
    OUTPUTS:
    - probability: The probability of a random variable C={'c','nc'} given a tag.
    """
    capital_count={}
    for tag in bag_of_tag:
        capital_count[tag]={'c':0,'nc':0}
    for word in bag_of_words:
        if (word[0]>='A'and word[0]<='Z'):
            for tag in bag_of_words[word]:
                capital_count[tag]['c']+=bag_of_words[word][tag]
        else:
            for tag in bag_of_words[word]:
                capital_count[tag]['nc']+=bag_of_words[word][tag]
    probability={}
    for tag in capital_count:
        total=sum(capital_count[tag].values())+len(capital_count[tag])*0.01
        probability[tag]={}
        for x in capital_count[tag]:
            probability[tag].update({x:(capital_count[tag][x]+0.01)/total})
    return probability
#%%
def tag_suffix(bag_of_words,bag_of_tag):
    """
    Calculate the probability of P(suffix|tag). suffix is a random variable S={'ed','ing','er','est','s','es','ly','able','en','digit','others'}
        
    INPUTS:
    - bag_of_words:A dictionary. Key is the word, Value is another dictionary which key is POS tag and value is tag count.
    - bag_of_tag:A dictionary. Key is the POS tag, Value is the tag count.
    
    OUTPUTS:
    - probability: The probability of a random variable S given a tag.
    """
    tag_suffix_count={}
    for tag in bag_of_tag:
        tag_suffix_count[tag]={'ed':0,'ing':0,'er':0,'est':0,'s':0,'es':0,'ly':0,'able':0,'en':0,'digit':0,'others':0}
    for word in bag_of_words:
        suffix=getSuffix(word)
        for tag in bag_of_words[word]:
            tag_suffix_count[tag][suffix]+=bag_of_words[word][tag]
    probability={}
    for tag in tag_suffix_count:
        total=sum(tag_suffix_count[tag].values())+len(tag_suffix_count[tag])*0.01
        probability[tag]={}
        for x in tag_suffix_count[tag]:
            probability[tag].update({x:(tag_suffix_count[tag][x]+0.01)/total})
    return probability
#%%
def getSuffix(word):
    """
    Get the suffix of a given word
        
    INPUTS:
    - word:A single string token.
    
    OUTPUTS:
    - suffix: The suffix of a word under the random variable S.
    """
    suffix='others'
    if(isNum(word)):
        suffix='digit'
    if(word.endswith('ed')):
        suffix='ed'
    if(word.endswith('ing')):
        suffix='ing'
    if(word.endswith('er')):
        suffix='er'
    if(word.endswith('est')):
        suffix='est'    
    if(word.endswith('s')):
        suffix='s'
    if(word.endswith('es')):
        suffix='es'
    if(word.endswith('ly')):
        suffix='ly'
    if(word.endswith('able')):
        suffix='able'
    if(word.endswith('en')):
        suffix='en'
    return suffix
#%% Witten- Bell
def wittenBell(sourceDict):
    """
    Witten-Bell smoothing algorithm
        
    INPUTS:
    - sourceDict:The target smoothing dictionary.
    
    OUTPUTS:
    - probability: The smoothed probability distribution.
    """
    probability={}
    for pretag in sourceDict:
        V=len(sourceDict[pretag])
        T=sum(1 for tag in sourceDict[pretag] if sourceDict[pretag][tag]>0)
        Z=V-T
        total=sum(sourceDict[pretag].values())
        probability[pretag]={}
        for curtag in sourceDict[pretag]:
            if(sourceDict[pretag][curtag]!=0):
                probability[pretag][curtag]=sourceDict[pretag][curtag]/(total+T)
            else:
                probability[pretag][curtag]=T/(Z*(total+T))
    return probability
#%% processing the file
train_file_name = sys.argv[1]
dev_file_name=sys.argv[2]
output_file_name=sys.argv[3]
train_file=open(train_file_name,'r')
train_view=train_file.readlines()
train_file.close()
dev_file=open(dev_file_name,'r')
dev_view=dev_file.readlines()
dev_file.close()
output_file=open(output_file_name,'w')
print "Loading file finish"
#%% HMM basic variables
tag_words,bag_of_tag,bag_of_words=train_preprocess(train_view)
tag_transition_count= tagTransitionCount(bag_of_tag, train_view)
tag_words_prob=tagWordProb(tag_words)
tag_transition_prob=wittenBell(tag_transition_count)
#%% Handling the unknown word
unknown_word_tag_prob=tag_unknown(bag_of_words,bag_of_tag)
tag_capital_prob=tag_capital(bag_of_words,bag_of_tag)
tag_suffix_prob=tag_suffix(bag_of_words,bag_of_tag)
#%% Dump into pickle
pickle.dump(bag_of_tag,output_file)
pickle.dump(bag_of_words,output_file)
pickle.dump(tag_words_prob,output_file)
pickle.dump(tag_transition_prob,output_file)
pickle.dump(unknown_word_tag_prob,output_file)
pickle.dump(tag_capital_prob,output_file)
pickle.dump(tag_suffix_prob,output_file)
print "model build finished"
