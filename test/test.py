import os
import sys
path = '/'.join([i for i in sys.path[0].split('\\')[:-1]])
path = path + '/src'
sys.path.append(path)
import LDA  
import matplotlib.pyplot as plt
#import utils

def printAttr(lda):
    print('K',lda.K)
    print('_uniqTermSet',lda._uniqTermSet)
    print('docsSize',lda._docNum)
    print('termSize',lda._termNum)
    print('Z ini:',lda.Z)
    print('docTopic ini',lda._docTopic)  ##4 doc,2topic
    print('lda.termTopic',lda._termTopic) 
    print('lda.Phi',lda.Phi)
    print('lda.Theta',lda.Theta)

if __name__ == "__main__":
    corpus = [
        "With all of the critical success Downey had experienced throughout his career, he had not appeared in a blockbuster film. That changed in 2008 when Downey starred in two critically and commercially successful films, Iron Man and Tropic Thunder. In the article Ben Stiller wrote for Downey's entry in the 2008 edition of The Time 100, he offered an observation on Downey's commercially successful summer at the box office.",
        "On June 14, 2010, Downey and his wife Susan opened their own production company called Team Downey. Their first project was The Judge.",
        "Robert John Downey Jr. is an American actor, producer, and singer. His career has been characterized by critical and popular success in his youth, followed by a period of substance abuse and legal troubles, before a resurgence of commercial success in middle age.",
        "In 2008, Downey was named by Time magazine among the 100 most influential people in the world, and from 2013 to 2015, he was listed by Forbes as Hollywood's highest-paid actor. His films have grossed over $14.4 billion worldwide, making him the second highest-grossing box-office star of all time."
        ]

    X = [i.split(' ') for i in corpus]
    lda = LDA.LDA()
    lda.fit(X)

    printAttr(lda)

    #fig,ax= lda.plotDocTopicDist(2)

    #fig,ax = lda.plotTermTopicDist(2)

    #fig,ax = lda.plotTopicTermDist(1)
    plt.show()