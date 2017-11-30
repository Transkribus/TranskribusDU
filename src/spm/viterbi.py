"""
    https://github.com/phvu/misc/tree/master/viterbi
"""

import numpy as np
 
class Decoder(object):
    def __init__(self, initialProb, transProb, obsProb):
        self.N = initialProb.shape[0]   #number of states
        self.initialProb = initialProb
        self.transProb = transProb
        self.obsProb = obsProb
        assert self.initialProb.shape == (self.N, 1)
        assert self.transProb.shape == (self.N, self.N)
        assert self.obsProb.shape[0] == self.N
 
    def Obs(self, obs):
        return self.obsProb[:, obs, None]
 
    def Decode(self, obs):
        trellis = np.zeros((self.N, len(obs)),dtype=np.float64)
        backpt = np.ones((self.N, len(obs)), dtype=np.int32) * -1
 
        # initialization
        trellis[:, 0] = np.squeeze(self.initialProb * self.Obs(obs[0]))
 
        for t in xrange(1, len(obs)):
            trellis[:, t] = (trellis[:, t-1, None].dot(self.Obs(obs[t]).T) * self.transProb).max(0)
            if (trellis[:, t] < 1e-100).all():
                trellis[:, t] =  trellis[:, t] * 1e+100
            backpt[:, t] = (np.tile(trellis[:, t-1, None], [1, self.N]) * self.transProb).argmax(0)
        
#         print trellis
        #         print trellis[:,-1].max()
#         print trellis[:,-1]*1e+100
#         print (trellis[:, -1] < 1e-100).all()
        tokens = [trellis[:, -1].argmax()]
        for i in xrange(len(obs)-1, 0, -1):
            tokens.append(backpt[tokens[-1], i])
        return tokens[::-1],  trellis[:, -1].max(0)