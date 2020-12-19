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

# A class for evaluating POS-tagged data
class Eval:
    ################################
    #intput:                       #
    #    goldFile: string          #
    #    testFile: string          #
    #output: None                  #
    ################################
    def __init__(self, goldFile, testFile):
        print("Your task is to implement an evaluation program for POS tagging")
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
        
    ################################
    #intput: None                  #
    #output: float                 #
    ################################
    def getTokenAccuracy(self):
        print("Return the percentage of correctly-labeled tokens")
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
        print("Return the percentage of sentences where every word is correctly labeled")
        return 1.0

    ################################
    #intput:                       #
    #    outFile: string           #
    #output: None                  #
    ################################
    def writeConfusionMatrix(self, outFile):
        print("Write a confusion matrix to outFile; elements in the matrix can be frequencies (you don't need to normalize)")

    ################################
    #intput:                       #
    #    tagTi: string             #
    #output: float                 #
    ################################
    def getPrecision(self, tagTi):
        print("Return the tagger's precision when predicting tag t_i")
        return 1.0

    ################################
    #intput:                       #
    #    tagTi: string             #
    #output: float                 #
    ################################
    # Return the tagger's recall on gold tag t_j
    def getRecall(self, tagTj):
        print("Return the tagger's recall for correctly predicting gold tag t_j")
        return 1.0


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
