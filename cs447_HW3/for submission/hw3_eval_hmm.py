########################################
## CS447 Natural Language Processing  ##
##           Homework 2               ##
##       Julia Hockenmaier            ##
##       juliahmr@illnois.edu         ##
########################################
##
## Part 1:
## Evaluate the output of your bigram HMM POS tagger
##
import os.path
import sys
from operator import itemgetter
from collections import defaultdict
from collections import Counter

# A class for evaluating POS-tagged data
class Eval:
    ################################
    #intput:                       #
    #    goldFile: string          #
    #    testFile: string          #
    #output: None                  #
    ################################
    def __init__(self, goldFile, testFile):
        self.confusionCnt = defaultdict(lambda : defaultdict(lambda:0))
        self.tags = []

        gold = open(goldFile,'r')
        self.gold_sens = []
        for line in gold:
            raw = line.split()
            sentence = []
            for token in raw:
                sentence.append(token.split('_')[1])
            self.gold_sens.append(sentence)

        test = open(testFile,'r')
        self.test_sens = []
        for line in test:
            raw = line.split()
            sentence = []
            for token in raw:
                sentence.append(token.split('_')[1])
            self.test_sens.append(sentence)

        for sen_idx in range(len(self.gold_sens)):
            sen = self.gold_sens[sen_idx]
            for tag_idx in range(len(sen)):
                tag_gold = self.gold_sens[sen_idx][tag_idx]
                tag_test = self.test_sens[sen_idx][tag_idx]
                self.confusionCnt[tag_gold][tag_test] += 1

        self.tags = list(self.confusionCnt.keys())
        

        
    ################################
    #intput: None                  #
    #output: float                 #
    ################################
    def getTokenAccuracy(self):
        # print("Return the percentage of correctly-labeled tokens")
        acc = 0
        total = 0
        for sen_idx in range(len(self.gold_sens)):
            sen = self.gold_sens[sen_idx]
            for tag_idx in range(len(sen)):
                tag_gold = self.gold_sens[sen_idx][tag_idx]
                tag_test = self.test_sens[sen_idx][tag_idx]
                if tag_gold == tag_test:
                    acc+=1
            total+=len(sen)
        return acc/total

    ################################
    #intput: None                  #
    #output: float                 #
    ################################
    def getSentenceAccuracy(self):
        # print("Return the percentage of sentences where every word is correctly labeled")
        acc = 0
        total = 0
        for sen_idx in range(len(self.gold_sens)):
            correct = True
            sen = self.gold_sens[sen_idx]
            for tag_idx in range(len(sen)):
                tag_gold = self.gold_sens[sen_idx][tag_idx]
                tag_test = self.test_sens[sen_idx][tag_idx]
                if tag_gold != tag_test:
                    correct = False
            if correct:
                acc += 1
            total += 1                 
        return acc/total

    ################################
    #intput:                       #
    #    outFile: string           #
    #output: None                  #
    ################################
    def writeConfusionMatrix(self, outFile):
        # print("Write a confusion matrix to outFile; elements in the matrix can be frequencies (you don't need to normalize)")

        f = open(outFile, 'w')
        for tag in self.tags:
            f.write('\t{}'.format(tag))
        f.write('\n')
        for tag1 in self.tags:
            f.write(tag1)
            for tag2 in self.tags:
                f.write('\t{}'.format(self.confusionCnt[tag1][tag2]))
            f.write('\n')




    ################################
    #intput:                       #
    #    tagTi: string             #
    #output: float                 #
    ################################
    def getPrecision(self, tagTi):


        total = 0
        for tag in self.tags:
            total += self.confusionCnt[tag][tagTi]
        acc = self.confusionCnt[tagTi][tagTi]
        return acc/total

    ################################
    #intput:                       #
    #    tagTi: string             #
    #output: float                 #
    ################################
    # Return the tagger's recall on gold tag t_j
    def getRecall(self, tagTj):


        total = sum(self.confusionCnt[tagTj].values())
        acc = self.confusionCnt[tagTj][tagTj]
        return acc/total



if __name__ == "__main__":
    # Pass in the gold and test POS-tagged data as arguments
    if len(sys.argv) < 2:
        print("Call hw2_eval_hmm.py with two arguments: gold.txt and out.txt")
    else:
        gold = sys.argv[1]
        test = sys.argv[2]
        # You need to implement the evaluation class
        eval = Eval(gold, test)
        # Calculate accuracy (sentence and token level)
        print("Token accuracy: ", eval.getTokenAccuracy())
        print("Sentence accuracy: ", eval.getSentenceAccuracy())
        # Calculate recall and precision
        print("Recall on tag NNP: ", eval.getPrecision('NNP'))
        print("Precision for tag NNP: ", eval.getRecall('NNP'))
        # Write a confusion matrix
        eval.writeConfusionMatrix("confusion_matrix.txt")
