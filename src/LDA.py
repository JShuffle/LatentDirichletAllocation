from scipy.stats import multinomial

import numpy as np
import pandas as pd
import utils
# TODO:return the assignment result.
class LDA():
    """
    Latent Dirichlet Allocation implementation based on collapsed Gibbs sampling.
    """
    def __init__(self,alphas=0.1,betas=0.1,K=2,iteration=100):
        # TODO:
        # add to configure.
        # alphas/betas expand to vector
        # verbose
        """
        Parameters
        ----------
        alphas:int
            prior parameters of doc topic distribution(i.e. dirichlet prior of multinomial distribution). 
        
        betas:int
            prior parameters of term topic distribution(i.e. dirichlet prior of multinomial distribution).
        
        iteration:int
            Number of rounds of collapsd Gibbs sampling.
        """
        self.config = {}
        self.alphas = alphas
        self.betas = betas
        self.K = K
        self.itr = iteration

    def _check_params(self):
        pass

    def _createUniqTermSet(self):
        self._uniqTermSet = []
        for doc in self.X:
            for term in doc:
                if term not in self._uniqTermSet:
                    self._uniqTermSet.append(term)
        return self._uniqTermSet

    def _initializer(self):
        """
        Initialize the paramaters and variables:
        
        _uniqTermSet:list
            Vx1 vector, the set of terms in the training docs.

        _termNum:int
            numbers of unique term set.

        _docNum:int
            numbers of docs.

        _docsSize:list
            Nx1 vector, the number of terms in each doc.

        Z:(nested)list
            Z_ij(Z[i][j]) refers to the topic of term in i-th doc, j-th position(in current iteration round).

        _docTopic:ndarray
            Nxk array, docTopic[i][j] refers to term numbers belongs to jth topic in ith doc. 
        
        _termTopic:ndarray
            Vxk array, termTopic[i][j] refers to numbers of ith word(over all docs) in jth topic.
        
        _termCountsInTopic:list
             Kx1 vector, termCounsinTopic[i] refers to numbers of term(over all docs) in topic[i]. 
        
        Phi:ndarray
            Vxk matrix of term topic distribution, Phi[i][j] refers to the likelyhood of ith term belongs to j-th topic.

        Theta:ndarray
            Nxk matrix of doc topic distribution, Theta[i][j] refers to the likelyhood of ith doc belongs to j-th topic.
        """

        # uniq set 
        self._uniqTermSet = self._createUniqTermSet()
        #size of unique term number
        self._termNum = len(self._uniqTermSet)


        ### initialize Z matrix        
        self._docNum = self.getDocNum(self.X)
        self._docsSize = [] # vector Nx1
        for i in range(self._docNum):
            term_num_in_ith_doc = self.getTermNum(self.X[i])
            self._docsSize.append(term_num_in_ith_doc)
        maxDocSize = np.max(self._docsSize)
        
        # Z_ij(Z[i][j]) refer to term of i-th doc, j-th position.
        # since shape is not structural, convert to list would be more convenient.
        self.Z = np.zeros([self._docNum,maxDocSize],dtype=int)
        self.Z = self.Z.tolist()

        ### initialize saver
        # Nxk array, docTopic[i][j]: term number of jth topic in ith doc. 
        self._docTopic = np.zeros([self._docNum,self.K],dtype=int) 
        
        # Vxk array, termTopic[i][j]: number of ith word in jth topic.
        self._termTopic = np.zeros([self._termNum,self.K],dtype=int)

        # Kx1 vector, termCounsinTopic[i]: number of terms in topic[i]. 
        self._termCountsInTopic = [0 for i in range(self.K)]

        ### initialize topic distribution matrix learned by algorithms.
        # Vxk matrix, Phi[i][j]: likelyhood of ith term belongs to j-th topic
        self.Phi = np.zeros([self._termNum,self.K],dtype=float)
        # Nxk matrix, Theta[i][j]: likelyhood of ith doc belongs to j-th topic
        self.Theta = np.zeros([self._docNum,self.K],dtype=float) 

        # Z:list
        docNum = self.getDocNum(self.X)
        prior = 1/self.K
        for i in range(docNum):
            termNum = self.getTermNum(self.X[i])
            #print(termNum)
            self.Z[i] = self.Z[i][:termNum]
            for j in range(termNum):
                # random initialize with topic 0,1,2,...,K-1
                z_ij = multinomial.rvs(1,p=[prior for i in range(self.K)])
                # get the exact topic number by count the position, since index starts from 0.
                z_ij = np.where(z_ij==1)[0][0]
                self.Z[i][j] = int(z_ij)
                #print(self.Z[i][j])

        # docTopic
        for i,doc in enumerate(self.Z):
            for j,term_topic in enumerate(doc):
                self._docTopic[i][term_topic] +=1

        # termTopic
        for i,doc in enumerate(self.X):
            for j,term in enumerate(doc): # self.X[i][j]: term of ith doc,jth word.
                term_topic = self.Z[i][j]
                # get term rank in uniqTermSet
                term_rank = self._uniqTermSet.index(term)
                self._termTopic[term_rank][term_topic] +=1
        
        # termCountsInTopic
        for i,doc in enumerate(self.Z):
            for j,term_topic in enumerate(doc):
                self._termCountsInTopic[term_topic] +=1


    def _sampleTermTopic(self,
                        doc_rank,
                        term_cur_topic,
                        numTermsInDoc,
                        cur_topic_num_in_cur_term,
                        term_counts_in_cur_topic,
                        ):
        """
        Sampling a new topic of term from fully conditional multinomial distribution.
        First calculate the probability of belonging to each topic(for k in range(self.K)), 
        then renormalize and sample based on the normalized probability(category distribution).
        
        Parameters
        ----------
        doc_rank:
            order of doc from which the current term comes from.

        term_cur_topic:
            current topic of current term.

        numTermsInDoc:
            numbers of term in current doc.

        cur_topic_num_in_cur_term:
            numbers of currently assigned topics under the current term

        term_counts_in_cur_topic:
            numbers of term belong to current topic over all docs

        Returns
        -------
        z_new:
            a newly assigned topic of current term.
        """

        probs=[]
        for k in range(self.K):
            # [alphas_k + (n_tk - 1)]/[K*alphas + (n_t)]
            # n_tk: term number belongs to topic k in t-th doc(minus current term).
            n_tk = self._docTopic[doc_rank][term_cur_topic] -1
            # n_t: numbers of term in t-th doc(minus current term).
            n_t = numTermsInDoc -1
            terms_1st = (self.alphas + n_tk) / (self.K * self.alphas + n_t) 

            # [betas_wv + n_wv\ts,u]/[V*betas + (n_u)]
            # n_wv\ts,u: numbers of currently assigned topics under the current term(minus current term). 
            n_wv_ts_u = cur_topic_num_in_cur_term - 1
            # numbers of term belong to current topic over all docs(minus current term).
            n_u = term_counts_in_cur_topic -1
            terms_2nd = (self.betas + n_wv_ts_u) / (self._termNum * self.betas + n_u)

            # P(Z=k|·)
            p_topic_k = terms_1st * terms_2nd 
            probs.append(p_topic_k)
        
        # renormalize
        probs_norm = [i/sum(probs) for i in probs]
        # sample from category distribution
        z_new = multinomial.rvs(1,p = probs_norm)
        z_new = np.where(z_new==1)[0][0]
        return z_new

    def _runCollapsedGibbs(self):
        """
        Run the sampling process based on collapsed Gibbs.
        """
        for t in range(self.itr): # round of mcmc
            for i,doc in enumerate(self.X): 
                for j,term in enumerate(doc):
                    # X[i][j]: jth term in ith doc.
                    doc_rank = i
                    term_cur_topic = self.Z[i][j]
                    term_rank = self._uniqTermSet.index(term)
                    numTermsInDoc = self.getTermNum(doc) # n_t
                    cur_topic_num_in_cur_term = self._termTopic[term_rank][term_cur_topic]    # n_wv\ts,u
                    term_counts_in_cur_topic = self._termCountsInTopic[term_cur_topic] #n_u
                    
                    z_new = self._sampleTermTopic(
                                                doc_rank,
                                                term_cur_topic,
                                                numTermsInDoc,
                                                cur_topic_num_in_cur_term,
                                                term_counts_in_cur_topic)
                    self.Z[i][j] = z_new

    def _updataParams(self):
        """
        Now that we get the final topic assignment of terms(i.e. Z), the last step we need to do 
        is to updata related parameters: Phi:term topic distribution and Theta:doc topic distribution.
        """
        ### Phi
        for i,doc in enumerate(self.X):
            for j,term in enumerate(doc): # self.X[i][j]: term of ith doc,jth word.
                term_topic = self.Z[i][j]
                # get term rank in uniqTermSet
                term_rank = self._uniqTermSet.index(term)
                self.Phi[term_rank][term_topic] +=1
        # rescale by topic
        for k in range(self.K):
            kth_topic_sum = sum([x[k] for x in self.Phi])
            for j,term in enumerate(self.Phi):
                self.Phi[j][k] = self.Phi[j][k]/kth_topic_sum

        ### Theta
        for i,doc in enumerate(self.Z):
            for j,term_topic in enumerate(doc):
                self.Theta[i][term_topic] +=1
        # rescale by doc
        for i,doc in enumerate(self.Theta):
            doc_sum = sum(doc)
            for j in range(len(doc)):
                #print(doc_sum)
                #print(self.Theta[i][j])
                self.Theta[i][j] = self.Theta[i][j]/doc_sum

    def _assignTopicForDoc(self):
        pass
    def fit(self,X):
        """
        Fitting by running each steps.
        Firstly, initialize parameters.
        Secondly, run collapsed Gibbs sampling.
        Thirdly, update the learned parameters from MCMC.
        Finally, assign topic for each doc based on its greatest likelihood.
        Parameters
        ----------        
        X:ndarray
            2d array of docs(X[i][j]: j-th term in i-th docs).

        """
        self.X = X
        self._initializer()
        self._runCollapsedGibbs()
        self._updataParams()
        self._assignTopicForDoc()

    def printConfig(self):
        pass

    def getDocAssign(self,ith_doc=None):
        """
        Return the topic with the greatest likelihood of i-th doc.
        """

        self.Theta[i]
    
    def getDocNum(self,X):
        return len(X)

    def getTermNum(self,X,ndim=1):
        size = len(X)
        if ndim==1:
            return size
        else:
            num=0
            for i in range(size):
                for j in X[i]:
                    num+=1
            return num

    def getUniqueTerm(self):
        return self._uniqTermSet

    def getTermTopicDist(self,ith_term=None,all=False):
        """
        Return the topic distribution under a given term.
        Parameters
        ----------           
        ith_term:int/str
            term index/term name.
        all:Boolean
            If True, return the distribution dataframe of all term.
        """
        if all:
            return pd.DataFrame(self.Phi,index = self._uniqTermSet)
        else:
            if isinstance(ith_term,int):
                i = ith_term
                term = self._uniqTermSet[i]
            if isinstance(ith_term,str):
                try:
                    i = self._uniqTermSet.index(ith_term)
                    term = ith_term
                except ValueError:
                    err = ith_term + " is not in the Term Set,please check"
                    print(err)
            # renormalize.    
            sum_over_ith_term = sum(self.Phi[i])
            if sum_over_ith_term == 0:
                warn = "The term topic distribution has not been updated, please call fit function first."
                raise UserWarning(warn)

            renormed_dist = [i/sum_over_ith_term for i in self.Phi[i]]   
            return pd.DataFrame(renormed_dist,columns = [term])

    def getTopicTermDist(self,ith_topic=None,all=False):
        """
        Return the distribution of terms under a given topic index.
        Parameters
        ----------           
        ith_topic:int 
            topic index.
        all:Boolean
            If True, return the distribution dataframe of all topic.
        """
        if all:
            return pd.DataFrame(self.Phi,index = self._uniqTermSet)
        else:
            i = int(ith_topic)
            sum_over_ith_topic = np.round(sum([term[i] for term in self.Phi]))
            if sum_over_ith_topic < 1:
                warn = "The topic term distribution has not been updated, please call fit function first."
                raise UserWarning(warn)
            return pd.DataFrame([term[i] for term in self.Phi])
    

    def getDocTopicDist(self,ith_doc=None,all=False):
        """
        Return the topic distribution of doc under a given index.
        Parameters
        ----------      
        ith_doc:int 
            doc index.
        all:Boolean
            If True, return the distribution dataframe of all docs.
        """
        if all:
            return pd.DataFrame(self.Theta)
        else:
            i = int(ith_doc)
            if np.round(sum(self.Theta[i])) < 1:
                warn = "The doc topic distribution has not been updated, please call fit function first."
                raise UserWarning(warn)
            return pd.DataFrame(self.Theta[i])
    


    def plotTermTopicDist(self,ith_term=None,size=None,save=None,**kwargs):
        """
        Parameters
        ----------      
        ith_term:int/str
            passing to getTermTopicDist().
        size:tuple
            contains 2 element:horizontal figsize(fisrt) and vertical figsize(second).
        save:str
            path to save the fig
        """
        df = self.getTermTopicDist(ith_term)
        
        if size:
            h_size = size[0]
            v_size = size[1]
            if len(size) !=2:
                h_size = 5 
                v_size = 2 * len(df)
        else:
            h_size = 5  
            v_size = 2 * len(df)           
            
        series = df.iloc[:,0].sort_values()
        idx = series.index.values
        vals = series.values

        fig,ax = utils.barplot(size = (h_size,v_size),index = idx,values = vals,**kwargs)
        ax.set_ylabel("Topic")
        ax.set_xlabel("Probability")

        if save:
            fig.savefig(save,dpi=800)
        return fig,ax
    
    def plotTopicTermDist(self,ith_topic=None,size=None,save=None,**kwargs):
        """
        Parameters
        ----------
        ith_topic:int 
            passing to getTopicTermDist().
        size:tuple
            contains 2 element:horizontal figsize(fisrt) and vertical figsize(second).
        save:str
            path to save the fig.
        """
        df = self.getTopicTermDist(ith_topic)
        
        if size:
            h_size = size[0]
            v_size = size[1]
            if len(size) !=2:
                h_size = 5 
                v_size = 0.05 * len(df)  # since the term under topic may be too many to plot.
        else:
            h_size = 5  
            v_size = 0.05 * len(df)           
        
        series = df.iloc[:,0].sort_values()
        idx = series.index.values
        vals = series.values      

        fig,ax = utils.barplot(size = (h_size,v_size),index = idx,values = vals,**kwargs)
        ax.set_ylabel("Term")
        ax.set_xlabel("Probability")        

        if save:
            fig.savefig(save,dpi=800)
        return fig,ax

    def plotDocTopicDist(self,ith_doc=None,size=None,save=None,**kwargs):
        """
        Parameters
        ----------
        ith_doc:int 
            passing to getDocTopicDist().
        size:tuple
            contains 2 element:horizontal figsize(fisrt) and vertical figsize(second).
        save:str
            path to save the fig.
        """
        df = self.getDocTopicDist(ith_doc)
        
        if size:
            h_size = size[0]
            v_size = size[1]
            if len(size) !=2:
                h_size = 5 
                v_size = 2 * len(df)  # since the term under topic may be too many to plot.
        else:
            h_size = 5  
            v_size = 2 * len(df)           
        
        series = df.iloc[:,0].sort_values()
        idx = series.index.values
        vals = series.values      

        fig,ax = utils.barplot(size = (h_size,v_size),index = idx,values = vals,**kwargs)
        ax.set_ylabel("Topic")
        ax.set_xlabel("Probability")        

        if save:
            fig.savefig(save,dpi=800)
        fig.tight_layout()
        return fig,ax

def parser():
    pass


if __name__ == "__main__":
    pass


