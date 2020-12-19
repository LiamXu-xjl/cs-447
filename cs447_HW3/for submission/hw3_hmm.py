########################################
## CS447 Natural Language Processing  ##
##           Homework 2               ##
##       Julia Hockenmaier            ##
##       juliahmr@illnois.edu         ##
########################################
##
## Part 1:
## Train a bigram HMM for POS tagging
##
import os.path
import sys
from operator import itemgetter
from collections import defaultdict
from collections import Counter
from math import log
import numpy as np

# Unknown word token
UNK = 'UNK'

# Class that stores a word and tag together
class TaggedWord:
    def __init__(self, taggedString):
        parts = taggedString.split('_')
        self.word = parts[0]
        self.tag = parts[1]

# Class definition for a bigram HMM
class HMM:
### Helper file I/O methods ###
    ################################
    #intput:                       #
    #    inputFile: string         #
    #output: list                  #
    ################################
    # Reads a labeled data inputFile, and returns a nested list of sentences, where each sentence is a list of TaggedWord objects
    def readLabeledData(self, inputFile):
        if os.path.isfile(inputFile):
            file = open(inputFile, "r") # open the input file in read-only mode
            sens = []
            for line in file:
                raw = line.split()
                sentence = []
                for token in raw:
                    sentence.append(TaggedWord(token))
                sens.append(sentence) # append this list as an element to the list of sentences
            return sens
        else:
            print("Error: unlabeled data file %s does not exist" % inputFile)  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
            sys.exit() # exit the script

    ################################
    #intput:                       #
    #    inputFile: string         #
    #output: list                  #
    ################################
    # Reads an unlabeled data inputFile, and returns a nested list of sentences, where each sentence is a list of strings
    def readUnlabeledData(self, inputFile):
        if os.path.isfile(inputFile):
            file = open(inputFile, "r") # open the input file in read-only mode
            sens = []
            for line in file:
                sentence = line.split() # split the line into a list of words
                sens.append(sentence) # append this list as an element to the list of sentences
            return sens
        else:
            print("Error: unlabeled data file %s ddoes not exist" % inputFile)  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
            sys.exit() # exit the script
### End file I/O methods ###

    ################################
    #intput:                       #
    #    unknownWordThreshold: int #
    #output: None                  #
    ################################
    # Constructor
    def __init__(self, unknownWordThreshold=5):
        # Unknown word threshold, default value is 5 (words occuring fewer than 5 times should be treated as UNK)
        self.minFreq = unknownWordThreshold
        ### Initialize the rest of your data structures here ###
        self.freqCnt = Counter()
        self.tagCnt = Counter()
        self.bgCnt = Counter()
        self.initCnt = Counter()
        self.emissionCnt = defaultdict(lambda: defaultdict(lambda:0))

        self.transitionDict = defaultdict(lambda : 0)
        self.initDict = defaultdict(lambda : 0)
        self.emissionDict = defaultdict(lambda : 0)

        self.uniqueTags = []
        self.vocab = [UNK]

    ################################
    #intput:                       #
    #    trainFile: string         #
    #output: None                  #
    ################################
    # Given labeled corpus in trainFile, build the HMM distributions from the observed counts
    def train(self, trainFile):
        data = self.readLabeledData(trainFile) # data is a nested list of TaggedWords
        #print("Your first task is to train a bigram HMM tagger from an input file of POS-tagged text")
        
        for sen in data:
            for t_word in sen:
                # parsing
                tag = t_word.tag
                word = t_word.word
                
                # count the word frequency and tag frequency
                self.freqCnt[word] += 1 
                self.tagCnt[tag] += 1



                # build uniqueTags
                if tag not in self.uniqueTags:
                    self.uniqueTags.append(tag)


        for sen in data:
            for idx, t_word in enumerate(sen):
                word = t_word.word
                tag = t_word.tag
                
                # if a word is infrequent treat it as UNK (Piazza @401)
                if self.freqCnt[word] < self.minFreq:
                    word = UNK
                else:
                    # build uniqueWords
                    if word not in self.vocab:
                        self.vocab.append(word)

                # counting for emission
                self.emissionCnt[tag][word]+=1

                # counting for init 
                if idx == 0:
                    self.initCnt[tag]+=1

                # counting for bigram 
                if idx < len(sen)-1:
                    t_word_next = sen[idx+1]
                    tag_next = t_word_next.tag
                    bg = (tag, tag_next)
                    self.bgCnt[bg]+=1


        for tag1 in self.uniqueTags:  
            # calculating init prop Piazza @398
            if self.initCnt[tag1] > 0:
                self.initDict[tag1] = np.log(self.initCnt[tag1]/len(data))
            else:
                self.initDict[tag1] = -np.inf

            # transition prop
            tag1sum = 0
            for tag2 in self.uniqueTags:
                tag1sum += self.bgCnt[(tag1,tag2)]+1
            for tag2 in self.uniqueTags:
                # transition prop smoothing Piazza @382_f1
                self.transitionDict[(tag1,tag2)] = np.log((self.bgCnt[(tag1,tag2)]+1) / tag1sum)

        # emission prop Piazza @397
        for tag in self.uniqueTags:
            tagsum = self.tagCnt[tag]
            for word in self.emissionCnt[tag].keys():
                self.emissionDict[(tag,word)] = np.log(self.emissionCnt[tag][word]/tagsum)



    ################################
    #intput:                       #
    #     testFile: string         #
    #    outFile: string           #
    #output: None                  #
    ################################
    # Given an unlabeled corpus in testFile, output the Viterbi tag sequences as a labeled corpus in outFile
    def test(self, testFile, outFile):
        data = self.readUnlabeledData(testFile)
        f=open(outFile, 'w+')
        # for idx, sen in enumerate(data):
        for sen in data:
            # print('Tagging the {}th sentence out of {}'.format(idx,len(data)))
            vitTags = self.viterbi(sen)
            senString = ''
            for i in range(len(sen)):
                senString += sen[i]+"_"+vitTags[i]+" "
            print(senString.rstrip(), end="\n", file=f)

    ################################
    #intput:                       #
    #    words: list               #
    #output: list                  #
    ################################
    # Given a list of words, runs the Viterbi algorithm and returns a list containing the sequence of tags
    # that generates the word sequence with highest probability, according to this HMM
    def viterbi(self, words):
        #print("Your second task is to implement the Viterbi algorithm for the HMM tagger")
        # returns the list of Viterbi POS tags (strings)
        out = []
        tag_idx = []
        viterbi = defaultdict(lambda: defaultdict(lambda: 0))
        backpointer = defaultdict(lambda: defaultdict(lambda: 'NULL'))
        doneDict = defaultdict(lambda : defaultdict(lambda:False))

        for i in range(len(words)):
            wi = words[i]
            if wi not in self.vocab:
                wi = UNK
            for j in range(len(self.uniqueTags)):
                tj = self.uniqueTags[j]
                if self.emissionCnt[tj][wi] > 0:
                    # setting up the first column
                    if i == 0:
                        viterbi[j][0] = self.initDict[tj] + self.emissionDict[(tj,wi)]
                        backpointer[j][0] = 'NULL'
                    else:
                        # calculating the viterbi, set up the backpointer
                        temp_max = -np.inf
                        temp_prev = -1
                        for j_prev in range(len(self.uniqueTags)):
                            tj_prev = self.uniqueTags[j_prev]
                            if doneDict[j_prev][i-1] == False:
                                temp_viterbi = viterbi[j_prev][i-1] + self.transitionDict[(tj_prev,tj)] + self.emissionDict[(tj,wi)]
                                if temp_viterbi > temp_max:
                                    temp_max = temp_viterbi
                                    temp_prev = j_prev
                        viterbi[j][i] = temp_max
                        backpointer[j][i] = temp_prev
                else:
                    doneDict[j][i] = True
        
        
        # find the most possible ending tag in the last column in the trellis
        last_word_idx = len(words)-1
        curr_max = -np.inf
        last_tag_idx = -1
        for j in range(len(self.uniqueTags)):
            if doneDict[j][last_word_idx] == False:
                if viterbi[j][last_word_idx] > curr_max:
                    curr_max = viterbi[j][last_word_idx]
                    last_tag_idx = j
                    tag_idx.insert(0,last_tag_idx)
        last_tag = self.uniqueTags[last_tag_idx]
        out.insert(0,last_tag)
        last_word_idx -= 1
        
        # tracking along the backpointers
        while last_word_idx >= 0:
            last_tag_idx = backpointer[last_tag_idx][last_word_idx+1]
            last_tag = self.uniqueTags[last_tag_idx]
            out.insert(0,last_tag)
            tag_idx.insert(0,last_tag_idx)
            last_word_idx -= 1

        # print(tag_idx)
        return out 


        
        
if __name__ == "__main__":
    tagger = HMM()
    tagger.train('train.txt')
    tagger.test('test.txt', 'out.txt')
