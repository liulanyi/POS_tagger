# -*- coding: utf-8 -*-
"""
Created on Mon Oct 09 16:34:44 2017

@author: lanyi
"""

#%%
from __future__ import division
import numpy
import re
import math

#%%
'''
                                                10-folds cross validation
'''
#%%
def isNum(word):
    pattern=re.compile(r'^[0-9]+\.?[0-9]*')
    result=pattern.match(word)
    if result:
        return True
    else:
        return False
    
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

#%%
def train_preprocess(train_view):
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
    tag_transition={}
    tag_transition['<s>']={x:0 for x in bag_of_tag}
    tag_transition['<s>'].update({'</s>':0})
    for tag in bag_of_tag:
        tag_transition[tag]={x:0 for x in bag_of_tag}
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
    tagWordP={}
    for tag in tag_words:
        total=sum(tag_words[tag].values())
        tagWordP[tag]={word: tag_words[tag][word]/total for word in tag_words[tag]}
    return tagWordP

#%%
def tag_unknown(bag_of_words,bag_of_tag):
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
        probability[tag]={x:(capital_count[tag][x]+0.01)/total for x in capital_count[tag]}
    return probability
#%%
def tag_suffix(bag_of_words,bag_of_tag):
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
        probability[tag]={x:(tag_suffix_count[tag][x]+0.01)/total for x in tag_suffix_count[tag]}
    return probability
#%%
def getSuffix(word):
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
    
#%%
def addOneSmooth(lambd,sourceDict):
    probability={}
    for key in sourceDict:
        total=sum(sourceDict[key].values())+len(sourceDict[key])*lambd
        probability[key]={x:(sourceDict[key][x]+lambd)/total for x in sourceDict[key] }
    return probability

#%% Witten- Bell
def wittenBell(sourceDict):
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
            
#%%
def getTestResult(test_words_list,tag_word_probability,tag_transition_probability,bag_of_tag,unknown_word_tag_prob,tag_capital_prob,tag_suffix_prob):
    test_result=[]
    for line in test_words_list:
        test_temp_result_list=viterbi(line,tag_word_probability,tag_transition_probability,bag_of_tag,unknown_word_tag_prob,tag_capital_prob,tag_suffix_prob) 
        test_result.append(test_temp_result_list) 
    return test_result

#%%
def myviterbi(viterbi,path,word,tag_word_probability,tag_transition_probability,unknown_word_tag_prob,tag_capital_prob,tag_suffix_prob):
    cur_viterbi={}
    tempPath=path.copy()
    for curtag in tag_word_probability:
        if word in tag_word_probability[curtag]:
            max_v_value=-float('Inf')
            bestpretag='<s>'
            for pretag in viterbi:
                v_value=viterbi[pretag]+math.log(tag_transition_probability[pretag][curtag])+math.log(tag_word_probability[curtag][word])
                if(v_value>max_v_value):
                    max_v_value=v_value
                    bestpretag=pretag
            cur_viterbi[curtag]=max_v_value
            if(bestpretag!='<s>'):
                if(bestpretag in path):
                    tempPath[curtag]=[x for x in path[bestpretag]]
                    tempPath[curtag].append(bestpretag)
                else:
                    tempPath[curtag]=[bestpretag]
        else:
            max_v_value=-float('Inf')
            bestpretag='<s>'
            capital_mark='nc'
            suffix='others'
            if (word[0]>='A'and word[0]<='Z'):
                capital_mark='c'
            suffix=getSuffix(word)
            for pretag in viterbi:
                v_value=viterbi[pretag]+math.log(tag_transition_probability[pretag][curtag])+math.log(unknown_word_tag_prob[curtag])+math.log(tag_capital_prob[curtag][capital_mark])+math.log(tag_suffix_prob[curtag][suffix])+math.log(0.00001)
                if(v_value>max_v_value):
                    max_v_value=v_value
                    bestpretag=pretag
            cur_viterbi[curtag]=max_v_value
            if(bestpretag!='<s>'):
                if(bestpretag in path):
                    tempPath[curtag]=[x for x in path[bestpretag]]
                    tempPath[curtag].append(bestpretag)
                else:
                    tempPath[curtag]=[bestpretag]          
    return cur_viterbi,tempPath
            
    
#%%
'''
    Input ---- is the test words sequence and model statistics
    Output --- is the correspoding tag for the word sequence
'''
def viterbi(test_corpus,tag_word_probability,tag_transition_probability,bag_of_tag,unknown_word_tag_prob,tag_capital_prob,tag_suffix_prob):
    '''Initialize viterbi List'''
    viterbi={}
    viterbi['<s>']=0
    for key in bag_of_tag:
        viterbi[key]=-float('Inf')
    path={}
    
    '''
        Go through the viterbi algorithm for each word
        viterbi is the List to store the most possible tag
        Path is the List to store the a sequence of tag
    '''
    for word in test_corpus:
        viterbi,path=myviterbi(viterbi,path,word,tag_word_probability,tag_transition_probability,unknown_word_tag_prob,tag_capital_prob,tag_suffix_prob)
                      
    '''
        Final step of viterbi algorithm
        From the viterbi List, find the tag with the largest probability
        Retrieve the tag sequence from the path List.
    '''
    
    final_tag='</s>'
    max_p=-float('Inf')
    last_tag='<s>'
    for pretag in viterbi:
        temp=viterbi[pretag]+math.log(tag_transition_probability[pretag][final_tag])
        if (temp>max_p):
            max_p=temp
            last_tag=pretag      
    #last_tag=max(viterbi, key=viterbi.get)
    #print last_tag
    output_tag=[]
    if (path.get(last_tag)!=None):
        output_tag=[x for x in path.get(last_tag)]
    output_tag.append(last_tag)
    return output_tag     

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
for i in range(7,K+1):
    print "test number: ",i
    train_view,test_view=Kfold(data_view,K,i)
    print "10 fold seperation finish"
    #%% Tag_transition and word tag
    tag_words,bag_of_tag,bag_of_words=train_preprocess(train_view)
    tag_transition_count= tagTransitionCount(bag_of_tag, train_view)
    tag_words_prob=tagWordProb(tag_words)
    #tag_transition_prob=addOneSmooth(0.01,tag_transition_count)
    tag_transition_prob=wittenBell(tag_transition_count)

    #%%
    unknown_word_tag_prob=tag_unknown(bag_of_words,bag_of_tag)
    tag_capital_prob=tag_capital(bag_of_words,bag_of_tag)
    tag_suffix_prob=tag_suffix(bag_of_words,bag_of_tag)
    #%%
    test_words_list, test_true_tag=test_preprocess(test_view)
    #%%    
    result=getTestResult(test_words_list,tag_words_prob,tag_transition_prob,bag_of_tag,unknown_word_tag_prob,tag_capital_prob,tag_suffix_prob)

    #%%
    known_word_precision,unknown_word_precision,avgPrecision,known_wrong_tag,unknown_wrong_tag=getPrecision(result, test_true_tag, bag_of_words, test_words_list)                
                        
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

        

