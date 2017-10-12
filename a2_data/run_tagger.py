# -*- coding: utf-8 -*-
"""
Created on Mon Oct 09 13:33:25 2017

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
                                                        Run file
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
#%%
def getTestResult(test_words_list,tag_word_probability,tag_transition_probability,bag_of_tag,unknown_word_tag_prob,tag_capital_prob,tag_suffix_prob):
    """
    Get the predicted tag lists from the model and test corpus.
        
    INPUTS:
    - test_words_list:A list of test file which element is a list of sequences of words.
    - tag_word_probability: P(word|tag)
    - tag_transition_probability: P(tag_n|tag_p)
    - bag_of_tag: A dictionary of the POS tag.
    - unknown_word_tag_prob: P(unknown_word|tag)
    - tag_capital_prob: P(isCapital|tag)
    - tag_suffix_prob: P(suffix|tag)
    
    OUTPUTS:
    - test_result: A list of predicted tags
    """
    test_result=[]
    for line in test_words_list:
        test_temp_result_list=viterbi(line,tag_word_probability,tag_transition_probability,bag_of_tag,unknown_word_tag_prob,tag_capital_prob,tag_suffix_prob) 
        test_result.append(test_temp_result_list) 
    return test_result

#%%
def myviterbi(viterbi,path,word,tag_word_probability,tag_transition_probability,unknown_word_tag_prob,tag_capital_prob,tag_suffix_prob):
    """
    Update the viterbi array and path dictionary for a single word
        
    INPUTS:
    - viterbi:A dictionary which record the viterbi array value of the last word. Key is tag, value is the probability.
    - path: A dictionary which record the backtrace path
    - tag_word_probability: P(word|tag)
    - tag_transition_probability: P(tag_n|tag_p)
    - unknown_word_tag_prob: P(unknown_word|tag)
    - tag_capital_prob: P(isCapital|tag)
    - tag_suffix_prob: P(suffix|tag)
    
    OUTPUTS:
    - cur_viterbi: updated viterbi dictionary
    - tempPath: updated backtrace path
    """
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
def viterbi(test_corpus,tag_word_probability,tag_transition_probability,bag_of_tag,unknown_word_tag_prob,tag_capital_prob,tag_suffix_prob):
    """
    Initialize the viterbi and path dictionary and iteratively call the myviterbi function for each word
        
    INPUTS:
    - test_corpus:A list of words for a sentence
    - tag_word_probability: P(word|tag)
    - tag_transition_probability: P(tag_n|tag_p)
    - bag_of_tag: A dictionary of the POS tag.
    - unknown_word_tag_prob: P(unknown_word|tag)
    - tag_capital_prob: P(isCapital|tag)
    - tag_suffix_prob: P(suffix|tag)
    
    OUTPUTS:
    - output_tag : A list of predicted tag for a sentence
    """
    
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

#%% Calculate the precision
def getPrecision(test_value, true_value, bag_of_words, test_words_list):
    """
    Calculate the precision of the prediction
        
    INPUTS:
    - test_value: The predicted tag list
    - true_value: The true tag list
    - bag_of_words: A list of test file which element is a list of sequences of words.
    - test_words_list: A dictionary. Key is the word, Value is another dictionary which key is POS tag and value is tag count.

    
    OUTPUTS:
    - known_word_precision: The precision for the known words. C(correct_known_words_predict)/C(total_known_words)
    - unknown_word_precision： The precision for the unknown words. C(correct_unknown_words_predict)/C(total_unknown_words)
    - precision：The average precision for the words. C(correct_predict)/C(total_words)
    - known_wrong_tag: The true tag list which is predicted incorrectly for the known words.
    - unknown_wrong_tag: The true tag list which is predicted incorrectly for the unknown words.
    """
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
def test_preprocess(test_view):
    """
    Preprocess the test file
        
    INPUTS:
    - test_view: A list of sentences of test file

    OUTPUTS:
    - test_words_list: Tokenize each element for the list 
    """
    test_words_list=[]
    for sentence in test_view:
        tempWordList=[]
        for word in sentence.split():
            tempWordList.append(word)
        test_words_list.append(tempWordList)
    return test_words_list
#%%
def output_factory(test_words_list, test_tag_list):
    """
    Generate the output
        
    INPUTS:
    - test_words_list: The tokenized words list

    OUTPUTS:
    - test_tag_list: The output list
    """
    result=[]
    for i in range(len(test_words_list)):
        sentence=""
        for j in range(len(test_words_list[i])):
            corpus=test_words_list[i][j]+'/'+test_tag_list[i][j]+' '
            sentence+=corpus
        result.append(sentence)    
    return result

#%% processing the file
test_file_name = sys.argv[1]
model_file_name=sys.argv[2]
output_file_name=sys.argv[3]
test_file=open(test_file_name,'r')
test_view=test_file.readlines()
test_file.close()
model_file=open(model_file_name,'r')
bag_of_tag=pickle.load(model_file)
bag_of_words=pickle.load(model_file)
tag_words_prob=pickle.load(model_file)
tag_transition_prob=pickle.load(model_file)
unknown_word_tag_prob=pickle.load(model_file)
tag_capital_prob=pickle.load(model_file)
tag_suffix_prob=pickle.load(model_file)
model_file.close()
output_file=open(output_file_name,'w')
print "Model loading finished"
#%%
test_words_list=test_preprocess(test_view)
#%%
result=getTestResult(test_words_list,tag_words_prob,tag_transition_prob,bag_of_tag,unknown_word_tag_prob,tag_capital_prob,tag_suffix_prob)
#%%
tagged_result=output_factory(test_words_list, result)
#%% write into file
for item in tagged_result:
    output_file.write("%s\n" % item)
output_file.close()
print "Tagged results have been written to file"

