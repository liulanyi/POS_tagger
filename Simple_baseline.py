# -*- coding: utf-8 -*-
"""
Created on Thu Oct 05 22:37:13 2017

@author: lanyi
"""
#%%
from __future__ import division
import numpy
import re
#%%
'''
                                                A Simple baseline (Assign the tag with best likely)
'''

#%%
def isNum(word):
    pattern=re.compile(r'^[0-9]+\.?[0-9]*')
    result=pattern.match(word)
    if result:
        return True
    else:
        return False
    
#%%
def mostProbableTag(word):
    mostProTag='NN'
    if(word[0]>='A'and word[0]<='Z'):
        mostProTag='NNP'
    if(word.endswith('able') or word.endswith('ible') or word.endswith('al') or word.endswith('ful')):
        mostProTag='JJ'
    if(word.endswith('er')):
        mostProTag='JJR'
    if(word.endswith('est')):
        mostProTag='JJS'
    if(word.endswith('ed')):
        mostProTag='VBN'
    if(word.endswith('ly')):
        mostProTag='RB'
    if(isNum(word)):
        mostProTag='CD'
    return mostProTag
        


#%% Calculate the precision
def getPrecision(test_value, true_value, bag_of_words, test_words_list):
    known_word_precision=[]
    unknown_word_precision=[]
    known_wrong_tag=[]
    unknown_wrong_tag=[]
    diff=0
    total=0
    precision=0
    assert(len(test_value)==len(true_value))
    for i in range(len(true_value)):
        assert(len(test_value[i])==len(true_value[i]))
        known_diff=0
        unknown_diff=0
        known_total=0
        unknown_total=0
        k_wrong_tag=[]
        u_wrong_tag=[]
        for j in range(len(true_value[i])):
            total+=1
            if test_words_list[i][j] not in bag_of_words:
                unknown_total+=1
                if(true_value[i][j]!=test_value[i][j]):
                    unknown_diff+=1
                    diff+=1
                    u_wrong_tag.append(true_value[i][j])
            else:
                known_total+=1
                if(true_value[i][j]!=test_value[i][j]):
                    known_diff+=1
                    diff+=1
                    k_wrong_tag.append(true_value[i][j])
        if(known_total!=0):
            known_word_precision.append((known_total-known_diff)/known_total)
        if(unknown_total!=0):
            unknown_word_precision.append((unknown_total-unknown_diff)/unknown_total)
        known_wrong_tag.append(k_wrong_tag)
        unknown_wrong_tag.append(u_wrong_tag) 
    precision=(total-diff)/total                            
    return known_word_precision,unknown_word_precision,precision,known_wrong_tag,unknown_wrong_tag
#%%
def Kfold(dataset,k,testnum):    
    num=len(dataset)
    leng=num//k
    teststart=(testnum-1)*leng
    testend=teststart+leng
    if testend > num-1:
        testend = num-1
        
    test_view = dataset[teststart:testend]
    train_view = dataset[:teststart]
    train_view += dataset[testend:]
    
    return train_view,test_view
#%%
def test_preprocess(test_view):
    test_words_list=[]
    test_true_tag=[]
    for sentence in test_view:
        tempWordList=[]
        tempTagList=[]
        for corpus in sentence.split():
            word=corpus.split('/')[0]
            tag=corpus.split('/')[-1]
            tempWordList.append(word)
            tempTagList.append(tag)
        test_words_list.append(tempWordList)
        test_true_tag.append(tempTagList)
    return test_words_list, test_true_tag

#%% Reading file
train_file=open("a2_data/sents.train",'r')
data_view=train_file.readlines()
train_file.close()
print "Loading file finish"
#%%
avg=[]
k_p=[]
u_p=[]
K=10
for i in range(1,K+1):
    print "test number: ",i
    train_view,test_view=Kfold(data_view,K,i)
    print "10 fold seperation finish"
#%% Count C(tag,word) and C(word)
    words_tag_count={}
    for sentence in train_view:
        for wordTag in sentence.split():
            tag=wordTag.split('/')[-1]
            for word in wordTag.split('/')[:-1]:
                if word not in words_tag_count:
                    words_tag_count[word]={tag:1}
                else:
                    if tag not in words_tag_count[word]:
                        words_tag_count[word]={tag:1}
                    else:
                        words_tag_count[word][tag]+=1
    print "C(word,tag)", words_tag_count['to']
#%% Count the probability of P(tag|word)
    probability={}
    for word in words_tag_count:
        total=sum(words_tag_count[word].values())
        probability[word]={x:(words_tag_count[word][x])/total for x in words_tag_count[word] }
    print "P(tag|word)", probability['to']
    
#%%
    test_words_list, test_true_tag=test_preprocess(test_view)
    result=[]
    for sentence in test_words_list:
        temp_result=[]
        for word in sentence:
            if word in probability:
                temp_result.append(max(probability[word]))
            else:
                temp_result.append(mostProbableTag(word))
        result.append(temp_result)
#%%
    known_word_precision,unknown_word_precision,avgPrecision,known_wrong_tag,unknown_wrong_tag=getPrecision(result, test_true_tag, probability, test_words_list)
    
#%%
    k_word_precision=numpy.mean(known_word_precision)
    u_word_precision=numpy.mean(unknown_word_precision)
    print "Known Precision: ", k_word_precision
    print "Unknown Precision: ", u_word_precision
    print "Avg Precision: ",avgPrecision
    avg.append(avgPrecision)
    k_p.append(k_word_precision)
    u_p.append(u_word_precision)             

print "total know precision is", numpy.mean(k_p)  
print "total uknow precision is", numpy.mean(u_p)      
print "total average precision is", numpy.mean(avg)        




