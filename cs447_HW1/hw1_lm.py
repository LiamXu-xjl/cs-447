########################################
## CS447 Natural Language Processing  ##
##           Homework 1               ##
##       Julia Hockenmaier            ##
##       juliahmr@illnois.edu         ##
########################################
##
## Part 1:
## Develop a smoothed n-gram language model and evaluate it on a corpus
##
import os.path
import sys
import random
import numpy as np
from operator import itemgetter
from collections import defaultdict
from collections import Counter
#----------------------------------------
#  Data input 
#----------------------------------------

# Read a text file into a corpus (list of sentences (which in turn are lists of words))
# (taken from nested section of HW0)
def readFileToCorpus(f):
    """ Reads in the text file f which contains one sentence per line.
    """
    if os.path.isfile(f):
        file = open(f, "r") # open the input file in read-only mode
        i = 0 # this is just a counter to keep track of the sentence numbers
        corpus = [] # this will become a list of sentences
        print("Reading file ", f)
        for line in file:
            i += 1
            sentence = line.split() # split the line into a list of words
            #append this lis as an element to the list of sentences
            corpus.append(sentence)
            if i % 1000 == 0:
    	#print a status message: str(i) turns int i into a string
    	#so we can concatenate it
                sys.stderr.write("Reading sentence " + str(i) + "\n")
        #endif
    #endfor
        return corpus
    else:
    #ideally we would throw an exception here, but this will suffice
        print("Error: corpus file ", f, " does not exist")
        sys.exit() # exit the script
    #endif
#enddef


# Preprocess the corpus to help avoid sess the corpus to help avoid sparsity
def preprocess(corpus):
    #find all the rare words
    freqDict = defaultdict(int)
    for sen in corpus:
	    for word in sen:
	       freqDict[word] += 1
	#endfor
    #endfor

    #replace rare words with unk
    for sen in corpus:
        for i in range(0, len(sen)):
            word = sen[i]
            if freqDict[word] < 2:
                sen[i] = UNK
	    #endif
	#endfor
    #endfor

    #bookend the sentences with start and end tokens
    for sen in corpus:
        sen.insert(0, start)
        sen.append(end)
    #endfor
    
    return corpus
#enddef

def preprocessTest(vocab, corpus):
    #replace test words that were unseen in the training with unk
    for sen in corpus:
        for i in range(0, len(sen)):
            word = sen[i]
            if word not in vocab:
                sen[i] = UNK
	    #endif
	#endfor
    #endfor
    
    #bookend the sentences with start and end tokens
    for sen in corpus:
        sen.insert(0, start)
        sen.append(end)
    #endfor

    return corpus
#enddef

# Constants 
UNK = "UNK"     # Unknown word token
start = "<s>"   # Start-of-sentence token
end = "</s>"    # End-of-sentence-token


#--------------------------------------------------------------
# Language models and data structures
#--------------------------------------------------------------

# Parent class for the three language models you need to implement
class LanguageModel:
    # Initialize and train the model (ie, estimate the model's underlying probability
    # distribution from the training corpus)
    def __init__(self, corpus):
        print("""Your task is to implement five kinds of n-gram language models:
      a) an (unsmoothed) unigram model (UnigramModel)
      b) a unigram model smoothed using Laplace smoothing (SmoothedUnigramModel)
      c) an unsmoothed bigram model (BigramModel)
      """)
        self.P_dict = {}
    #enddef

    # Generate a sentence by drawing words according to the 
    # model's probability distribution
    # Note: think about how to set the length of the sentence 
    #in a principled way
    def generateSentence(self):
        print("Implement the generateSentence method in each subclass")
        return "mary had a little lamb ."
    #emddef

    # Given a sentence (sen), return the probability of 
    # that sentence under the model
    def getSentenceProbability(self, sen):
        print("Implement the getSentenceProbability method in each subclass")
        return 0.0
    #enddef

    # Given a corpus, calculate and return its perplexity 
    #(normalized inverse log probability)
    def getCorpusPerplexity(self, corpus):
        return 0.0
    #enddef

    # Given a file (filename) and the number of sentences, generate a list
    # of sentences and write each to file along with its model probability.
    # Note: you shouldn't need to change this method
    def generateSentencesToFile(self, numberOfSentences, filename):
        filePointer = open(filename, 'w+')
        for i in range(0,numberOfSentences):
            sen = self.generateSentence()
            prob = self.getSentenceProbability(sen)
            stringGenerated = str(prob) + " " + " ".join(sen) 
            filePointer.write(stringGenerated+'\n\n')
        filePointer.close()
	#endfor
    #enddef
#endclass

# Unigram language model
class UnigramModel(LanguageModel):
    def __init__(self, corpus):
        # print("Subtask: implement the unsmoothed unigram language model")
        LanguageModel.__init__(self,corpus)
        corpus_out = []
        for y in corpus:
            for x in y:
                if x == start:
                    continue
                corpus_out.append(x)
        cnt = Counter(corpus_out)
        s = sum(cnt.values())
        for word, freq in cnt.items():
            self.P_dict[word] = freq/s

    def generateSentence(self):
        out = [start]
        curr = start
        while curr != end:
            curr = random.choices(list(self.P_dict.keys()),weights=list(self.P_dict.values()))[0]
            if curr == start:
                continue
            out.append(curr)
        return out


    def getSentenceProbability(self, sen):
        out = 1
        sen = sen[1:]
        for word in sen:
            out*=self.P_dict[word]
        return out

    def getCorpusPerplexity(self, corpus):
        corpus_out = []
        for y in corpus:
            for x in y:
                if x == start:
                    continue
                corpus_out.append(x)
        N = len(corpus_out)
        return np.exp(-1/N*np.sum(np.log(list(map(lambda x: self.P_dict[x], corpus_out)))))


#Smoothed unigram language model (use laplace for smoothing)
class SmoothedUnigramModel(LanguageModel):
    def __init__(self, corpus):
        LanguageModel.__init__(self,corpus)
        corpus_out = []
        for y in corpus:
            for x in y:
                if x == start:
                    continue
                corpus_out.append(x)
        cnt = Counter(corpus_out)
        N = len(corpus_out)
        S = len(cnt.keys())
        for word, freq in cnt.items():
            self.P_dict[word] = (freq+1)/(N+S)

    def generateSentence(self):
        out = [start]
        curr = start
        while curr != end:
            curr = random.choices(list(self.P_dict.keys()),weights=list(self.P_dict.values()))[0]
            if curr == start:
                continue
            out.append(curr)
        return out


    def getSentenceProbability(self, sen):
        out = 1
        sen = sen[1:]
        for word in sen:
            out*=self.P_dict[word]
        return out

    def getCorpusPerplexity(self, corpus):
        corpus_out = []
        for y in corpus:
            for x in y:
                if x == start:
                    continue
                corpus_out.append(x)
        N = len(corpus_out)
        return np.exp(-1/N*np.sum(np.log(list(map(lambda x: self.P_dict[x], corpus_out)))))

# Unsmoothed bigram language model
class BigramModel(LanguageModel):
    def __init__(self, corpus):
        LanguageModel.__init__(self,corpus)

        corpus_out = []
        for y in corpus:
            for i in range(len(y)-1):
                x0 = y[i]
                x1 = y[i+1]
                corpus_out.append((x0,x1))
        cnt = Counter(corpus_out)

        corpus_out_single = []
        for y in corpus:
            for x in y:
                corpus_out_single.append(x)
        cnt_single = Counter(corpus_out_single)

        self.cnt = cnt
        # s = sum(cnt.values())
        for word_pair, freq in cnt.items():
            self.P_dict[word_pair] = np.log(freq) - np.log(cnt_single[word_pair[0]])

    def generateSentence(self):
        out = [start]
        curr = start
        while curr != end:
            # curr = random.choices(list(self.P_dict.keys()),weights=list(self.P_dict.values()))[0]
            possible_pair = [p for p in list(self.cnt.keys()) if p[0] == curr]
            possible_weight = [self.cnt[p] for p in possible_pair]
            chosen_pair = random.choices(possible_pair,weights=possible_weight)[0]
            curr = chosen_pair[1]
            out.append(curr)
        return out


    def getSentenceProbability(self, sen):
        out = 0
        unexpected = False
        # sen = sen[1:]
        for i in range(len(sen)-1):
            pair = (sen[i],sen[i+1])
            if pair not in self.P_dict.keys():
                unexpected = True
                break
            out+=self.P_dict[pair]
        if unexpected:
            out = 0.0
            return out
        out = np.exp(out)
        return out


    def getCorpusPerplexity(self, corpus):
        out = 0
        N = 0
        for sen in corpus:
            for i in range(len(sen)-1):
                pair = (sen[i],sen[i+1])
                if pair not in self.P_dict.keys():
                    return np.inf
                out+=self.P_dict[pair]
            N += len(sen)
        out = np.exp(-1/N * out)
        return out
            


#-------------------------------------------
# The main routine
#-------------------------------------------
if __name__ == "__main__":
    #read your corpora
    trainCorpus = readFileToCorpus('train.txt')
    trainCorpus = preprocess(trainCorpus)
    uni =   UnigramModel(trainCorpus)
    smooth = SmoothedUnigramModel(trainCorpus)
    bi = BigramModel(trainCorpus)

    uni.generateSentencesToFile(20,'unigram_output.txt')
    # smooth.generateSentencesToFile(20,'smooth_unigram_output.txt')
    # bi.generateSentencesToFile(20,'bigram_output.txt')

    posTestCorpus = readFileToCorpus('pos_test.txt')
    negTestCorpus = readFileToCorpus('neg_test.txt')

    vocab = set()
    for sen in trainCorpus:
        for word in sen:
            vocab.add(word)
    # Please write the code to create the vocab over here before the function preprocessTest
    print("""Task 0: create a vocabulary(collection of word types) for the train corpus""")
    posTestCorpus = preprocessTest(vocab, posTestCorpus)
    negTestCorpus = preprocessTest(vocab, negTestCorpus)

    print ("POSTIVE")
    print (uni.getCorpusPerplexity(posTestCorpus))
    print (smooth.getCorpusPerplexity(posTestCorpus))
    print (bi.getCorpusPerplexity(posTestCorpus))

    print ("NEGTIVE")
    print (uni.getCorpusPerplexity(negTestCorpus))
    print (smooth.getCorpusPerplexity(negTestCorpus))
    print (bi.getCorpusPerplexity(negTestCorpus))


