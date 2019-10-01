"""

Several function to deal with the notion of longest common subsequence of two string, or any sequence in general

If you only wnat to know about the maximal length of the longest common substring, look at:
- fastlcs(a,b,Dmax=None) compute the length of the longest common substring, possibly under the constrint of a maximum number of differences
or
fastlcsfun(a,b,cmpfun, Dmax=None) if you want to provide your own compare function

If you want to know what is the longest substring and also have the Shortest Edit Sequence, look at:
- class LCS(..)
    lcs( ..)
    orderedlcs(..)
    
    
 JL Meunier - September 2000, June 2008
"""

DEBUG = 0

import time, array

def matchLCS(perc, t1, t2):
    (s1, n1) = t1
    (s2, n2) = t2

    nmax = max(n1, n2)
    nmin = min(n1, n2)
    
    if nmax <=0: return False,0
    #cases that obviously fail
    if nmin < (nmax * perc / 100.0 ): return False, 0

    #LCS
    n = lcs(s1, s2)
    val = round(100.0 * n / nmax)
    return ((val >= perc), val)

    
def testlcs(self,X,Y,m,n):
            L = [[0 for x in range(n+1)] for x in range(m+1)]
           
            # Following steps build L[m+1][n+1] in bottom up fashion. Note
            # that L[i][j] contains length of LCS of X[0..i-1] and Y[0..j-1] 
            for i in range(m+1):
                for j in range(n+1):
                    if i == 0 or j == 0:
                        L[i][j] = 0
                    elif X[i-1] == Y[j-1]:
                        L[i][j] = L[i-1][j-1] + 1
                    else:
                        L[i][j] = max(L[i-1][j], L[i][j-1])
           
            # Following code is used to print LCS
            index = L[m][n]
           
            # Create a character array to store the lcs string
            lcs = [""] * (index+1)
            lcs[index] = ""
            lmapping = []
            # Start from the right-most-bottom-most corner and
            # one by one store characters in lcs[]
            i = m
            j = n
            while i > 0 and j > 0:
           
                # If current character in X[] and Y are same, then
                # current character is part of LCS
                if X[i-1] == Y[j-1]:
                    lcs[index-1] = X[i-1]
                    lmapping.append((i-1,j-1))
                    i-=1
                    j-=1
                    index-=1
           
                # If not same, then find the larger of two and
                # go in the direction of larger value
                elif L[i-1][j] > L[i][j-1]:
                    i-=1
                else:
                    j-=1
           
            lmapping.reverse()
            xx =[(X[x],Y[y]) for x,y in lmapping]
            return xx
        
#--------- LCS code
# Return the length of the longest common string of a and b.
def lcs(a, b):

    na, nb = len(a), len(b)

    #switch a and b if b is shorter
    if nb < na:
        a, na, b, nb = b, nb, a, na

    curRow = [0]*(na+1)

    for i in range(nb):
        prevRow, curRow = curRow, [0]*(na+1)
        for j in range(na):
            if b[i] == a[j]:
                curLcs = max(1+prevRow[j], prevRow[j+1], curRow[j])
            else:
                curLcs = max(prevRow[j+1], curRow[j])
            curRow[j+1] = curLcs
    print (curRow)
    return curRow[na] 

def fastlcs(a,b,Dmax=None):
    """
    return the length of the longest common substring or 0 if the maximum number of difference Dmax cannot be respected

    Implementation: see the excellent paper "An O(ND) Difference Algorithm and Its Variations" by EUGENE W. MYERS, 1986
         
    NOTE:
     let D be the minimal number of insertion or deletion that transform A into B
     let L be the length of a longest common substring
     we always have D = M + N - 2 * L 
    """
    N, M = len(a), len(b)
    if N+M == 0: return 0  #very special case...
    
    if Dmax == None: 
        Dmax = N + M #worse case
    else:
        Dmax = min(Dmax, M+N) #a larger value does not make sense!
    assert Dmax >= 0, "SOFWARE ERROR: Dmax must be a positive integer"
    sesLength = None
    W = [0] * (Dmax * 2 + 2)   #for i in -Dmax..Dmax, V[i] == W[i+Dmax)
    for D in range(0, Dmax+1):
        for k in range(-D, +D+1, 2):
            if k == -D or (k != D and W[k-1+Dmax] < W[k+1+Dmax]):    #k == -D or (k != D and V[k-1] < V[k+1])
                x = W[k+1+Dmax]                                                                #x = V[k+1]
            else:
                x = W[k-1+Dmax]+1                                                            #x = V[k-1]+1
            y = x - k
            while x < N and y < M and a[x] == b[y]:  #follow any snake
                x += 1
                y += 1
            W[k+Dmax] = x    # V[k] = x     #farstest reaching point with D edits
            if x >= N and y >= M:
                sesLength = D
                L = (M+N-D) / 2
                assert D == M+N-L-L, ("INTERNAL SOFWARE ERROR", M,N,D)
                return L
    return 0    


def fastlcsfun(a,b,cmpfun, Dmax=None):
    """
    Same, but with a specific comparison function (different than ==, but following the same convention(return 0 if equal))
    
    return the length of the longest common substring or 0 if the maximum number of difference Dmax cannot be respected

    Implementation: see the excellent paper "An O(ND) Difference Algorithm and Its Variations" by EUGENE W. MYERS, 1986
         
    NOTE:
     let D be the minimal number of insertion or deletion that transform A into B
     let L be the length of a longest common substring
     we always have D = M + N - 2 * L 
    """
    N, M = len(a), len(b)
    if N+M == 0: return 0  #very special case...
    
    if Dmax == None: 
        Dmax = N + M #worse case
    else:
        Dmax = min(Dmax, M+N) #a larger value does not make sense!
    assert Dmax >= 0, "SOFWARE ERROR: Dmax must be a positive integer"
    sesLength = None
    W = [0] * (Dmax * 2 + 2)   #for i in -Dmax..Dmax, V[i] == W[i+Dmax)
    for D in range(0, Dmax+1):
        for k in range(-D, +D+1, 2):
            if k == -D or (k != D and W[k-1+Dmax] < W[k+1+Dmax]):    #k == -D or (k != D and V[k-1] < V[k+1])
                x = W[k+1+Dmax]                                                                #x = V[k+1]
            else:
                x = W[k-1+Dmax]+1                                                            #x = V[k-1]+1
            y = x - k
            while x < N and y < M and cmpfun(a[x],b[y]) == 0:  #follow any snake
                x += 1
                y += 1
            W[k+Dmax] = x    # V[k] = x     #farstest reaching point with D edits
            if x >= N and y >= M:
                sesLength = D
                L = (M+N-D) / 2
                assert D == M+N-L-L, ("INTERNAL SOFWARE ERROR", M,N,D)
                return L
    return 0    


def lcs_length(a,b):
    """
    Compute the length of the longest common string. Very fast. JLM March 2016
    
    NOTE: I did not compare against fastlcs...
    """
    na, nb = len(a), len(b)
    if nb < na: a, na, b, nb = b, nb, a, na
    if na==0: return 0
    na1 = na+1
    curRow  = [0]*na1
    prevRow = [0]*na1
    range1a1 = range(1, na1)
    for i in range(nb):
        bi = b[i]
        prevRow, curRow = curRow, prevRow
        curRow[0] = 0
        curRowj = 0
        for j in range1a1:
            if bi == a[j-1]:
                curRowj = max(1+prevRow[j-1], prevRow[j], curRowj)
            else:
                curRowj = max(prevRow[j], curRowj)
            curRow[j] = curRowj
    return curRowj

#
# matches two items
# Default behavior
# return true if the items match
# 1 if item1 > item2
def match(item1, item2):
    return item1 == item2



class LCS:
    #
    #Get two list of strings, or characters,
    # or two strings
    # and possibly a match function, which return True if two item are identical
    def __init__(self, l1, l2, matchfun = match):
        self.l1 = l1
        self.l2 = l2
        self.matchfun = matchfun

    # Build the longest common sub-list
    # returns:
    # - the number of common items
    # - the number of items missing from l1
    # - the number of items missing from l2
    # - a summary list: [  (item, status), ...]
    #  where status is 0, 1, 2 for resp. ok, missing from l1, missing from l2
    # 
    #
    def lcs(self):
        t0 = time.time()

        #Find the best substring
        #The length MUST be computed prior to the LCS itself
        length = self.getLength()

        t1 = time.time()
        if DEBUG:
            print ("LCS length:", length)
            print ('   computed in %fs'%(t1-t0))

        #Build the best substring
        t = self.buildLCS()

        t2 = time.time()
        if DEBUG:
            print ('   lcs built in %fs'%(t2-t1))

        return t

       

    def orderedlcs(self, cmpfun=None):
        """
        Build the longest common sub-list for the (easy) case where list are ordered
        In this case, rather than using the matchfun, it uses the cmpfun, which works like the python cmp function.
        (   cmp(x, y)
            Compare the two objects x and y and return an integer according to the outcome. 
            The return value is negative if x < y, zero if x == y and strictly positive if x > y.
        )
        if cmpfun is None, we assume the matchfun is actually a compare function
     returns:
     - the number of common items
     - the number of items missing from l1
     - the number of items missing from l2
     - a summary list: [  (item, status), ...]
      where status is 0, 1, 2 for resp. ok, missing from l1, missing from l2
     
     *** WARNING *** if the lists are not ordered according to cmpfun, the results maight be crazy!!!
        """
        if not cmpfun:
            cmpfun = self.matchfun
            
        t0 = time.time()

        i1, i2 = 0,0
        nok, miss1, miss2 = 0,0,0
        lStatus = []
        
        i1max, i2max = len(self.l1), len(self.l2)
        try:
            e1, e2 = self.l1[i1], self.l2[i2]
            while True:
                ret = cmpfun(e1, e2)
                if ret < 0: #e1 < e2
                    #e1 missing from l2
                    miss1 += 1
                    lStatus.append( (e1, 2) )
                    i1 += 1
                    e1 = self.l1[i1]
                elif ret > 0:
                    miss2 += 1
                    lStatus.append( (e2, 1) )
                    i2 += 1
                    e2 = self.l2[i2]
                else: #e1 == e2
                    assert ret == 0
                    nok += 1
                    lStatus.append( (e1, 0) ) #let's put e1 in the status list
                    i1 += 1
                    i2 += 1
                    e1 = self.l1[i1]
                    e2 = self.l2[i2]
        except IndexError:
            pass
                
        #now any remaining elements from either list
        while i1 < i1max:
            lStatus.append( (self.l1[i1], 2) )
            i1 += 1
            miss1 += 1
        while i2 < i2max:
            lStatus.append( (self.l2[i2], 1) )
            i2 += 1            
            miss2 += 1

        t1 = time.time()
        if DEBUG:
            print ('   ordered lcs in %fs'%(t1-t0))

        return nok, miss2, miss1, lStatus
    
    #-----------------------------

    #Initialization of the internal memory
    def lcsinit(self):
        self.len1 = len(self.l1)
        self.len2 = len(self.l2)
        if DEBUG:
            print ('\tlength l1 :', self.len1)
            print ('\tlength l2 :', self.len2)

        #Matrix of mximal length of substring
        self.lenMat = []
        for ii in range(self.len1+1):
            l = array.array('l')
            for j in range(self.len2+1):
                l.append(-1)
            self.lenMat.append(l)


    #
    # Find the length of the longest common sub-lists
    # Build a memory of intermediary results
    # - the indice on list l1
    # - the indice on list l2
    # returns:
    # - the length of the longest sublist
    #
    def getLength(self):
        self.lcsinit()
        return self._getLength(0, 0)

    def _getLength(self, i1, i2):

        #Do we have already computed this case?
        n = self.lenMat[i1][i2]
        if n >= 0:
            return n
    
        #get the current item from each list, set a boolean if failure
        try:
            item1 = self.l1[i1]
            item2 = self.l2[i2]
            #compare their first item
            if apply(self.matchfun, (item1, item2)):   #MATCHING!
                n = 1 + self._getLength(i1+1, i2+1)
            else:                                      #item1 and item2 DIFFER
                #case A: item1 may be missing from l2 -> next item from l1
                nbcom_a = self._getLength(i1+1, i2)
                #case B: item2 may be missing from l1 -> next item from l2
                nbcom_b = self._getLength(i1, i2+1)
                n = max(nbcom_a, nbcom_b)
        except IndexError:
            #one of the list has no more item  => 0
            n = 0

        self.lenMat[i1][i2] = n
        return n

    #
    # Build the longest common sub-lists, using the memory
    # - the indice on list l1
    # - the indice on list l2
    # returns:
    # - the number of common items
    # - the number of items missing from l1
    # - the number of items missing from l2
    # - a summary list: [  (item, status), ...]
    #  where status is 0, 1, 2 for resp. ok, missing from l1, missing from l2
    # 
    #
    def buildLCS(self):
        return self._buildLCS(0, 0)

    def _buildLCS(self, i1, i2):
        nbcom = mf1 = mf2 = 0
        lsum = []

        try:
            while 1:
                item1 = self.l1[i1]
                item2 = self.l2[i2]

                if apply(self.matchfun, (item1, item2)):
                    nbcom = nbcom + 1
                    lsum.append( (item1, 0) )
                    i1 = i1 + 1
                    i2 = i2 + 1
                elif self.lenMat[i1+1][i2] >= self.lenMat[i1][i2+1]:
                    #skip item1
                    mf2 = mf2 + 1
                    i1 = i1 + 1
                    lsum.append( (item1, 2) )
                else:
                    #skip item2
                    mf1 = mf1 + 1
                    i2 = i2 + 1
                    lsum.append( (item2, 1) )
        except IndexError:
            try:
                while 1:
                    item1 = self.l1[i1]
                    mf2 = mf2 + 1
                    lsum.append( (item1, 2) )
                    i1 = i1 + 1
            except IndexError:
                pass
            try:
                while 1:
                    item2 = self.l2[i2]
                    mf1 = mf1 + 1
                    lsum.append( (item2, 1) )
                    i2 = i2 + 1
            except IndexError:
                pass

        return nbcom, mf1, mf2, lsum


    #debug utility
    def showlenMat(self):
        for i in range(self.len1 + 1):
            for j in range(self.len2 + 1):
                print ("%4d" % self.lenMat[i][j],)
            print()
        

            
       

    
if __name__ == "__main__":
    nTest, nErr = 0, 0
    def test(l1, l2, lStatus=None):
        global nTest, nErr
        nTest += 1
        print ('-----------------------------')
        print ('l1 ', l1)
        print ('l2 ', l2)

        o = LCS(l1, l2)
        ret = o.lcs()
        if ret[3] != lStatus: 
            print ("ERROR!!!",)
            nErr += 1
        else:
            print ("  OK  ",)
        print (ret)
    test( "abc", "ab", [('a', 0), ('b', 0), ('c', 2)])
    test( "", "" , [])
    test( "a", "" , [('a', 2)])
    test( "", "a" ,  [('a', 1)])


    test( [], [] , [])
    test( [1], [1]                 , [(1,0)])
    test( [1], [2]                 , [(1, 2), (2, 1)])
    test( [1, 2], [2]             , [(1, 2), (2, 0)])
    test( [1], [1, 2]             , [(1, 0), (2, 1)])
    test( [1,2], [1,3]             , [(1, 0), (2, 2), (3, 1)])
    test( [1,2,3,4], [1,2,4]     , [(1, 0), (2, 0), (3, 2), (4, 0)])


    test( [1, 2,   4, 7]
        , [1, 22, 24, 7]         , [(1, 0), (2, 2), (4, 2), (22, 1), (24, 1), (7, 0)])

    test( [1, 2, 32, 4, 7]
        , [1, 22, 3, 24, 7]     , [(1, 0), (2, 2), (32, 2), (4, 2), (22, 1), (3, 1), (24, 1), (7, 0)])

    test( [1, 4 , 25, 6 , 7, 28, 9 , 11 , 11, 11, 212, 22, 22]
        , [1, 22, 3 , 26, 7, 8 , 29, 211,211, 11, 12 , 22, 22] 
        , [(1, 0), (4, 2), (25, 2), (6, 2), (22, 1), (3, 1), (26, 1), (7, 0), (28, 2), (9, 2), (11, 2), (11, 2), (8, 1), (29, 1), (211, 1), (211, 1), (11, 0), (212, 2), (12, 1), (22, 0), (22, 0)])

    test( [1, 4 , 25, 6 , 7, 28, 9 , 11 , 11, 11, 212, 22, 22]
        , [1, 22, 3 , 26, 7, 8 , 29, 211,211, 11, 12 , 22, 22] 
        , [(1, 0), (4, 2), (25, 2), (6, 2), (22, 1), (3, 1), (26, 1), (7, 0), (28, 2), (9, 2), (11, 2), (11, 2), (8, 1), (29, 1), (211, 1), (211, 1), (11, 0), (212, 2), (12, 1), (22, 0), (22, 0)])

    test( [1, 2, 32, 4, 25, 6, 7, 28, 9, 11, 211, 11, 211, 11, 11, 212, 22, 22]
        , [1, 22, 3, 24, 5,26, 7, 8, 29, 211, 11, 211, 11, 211, 11, 12, 22, 22] 
        , [(1, 0), (2, 2), (32, 2), (4, 2), (25, 2), (6, 2), (22, 1), (3, 1), (24, 1), (5, 1), (26, 1), (7, 0), (28, 2), (9, 2), (11, 2), (8, 1), (29, 1), (211, 0), (11, 0), (211, 0), (11, 0), (211, 1), (11, 0), (212, 2), (12, 1), (22, 0), (22, 0)])
    
    print ("%d ERRORS on %d tests"%(nErr, nTest))
    
    def testOrdered(l1, l2, lStatus=None):
        global nTest, nErr
        nTest += 1
        print ('-----------------------------')
        print ('l1 ', l1)
        print ('l2 ', l2)

        o = LCS(l1, l2)
        ret = o.orderedlcs(cmp)
        if ret[3] != lStatus: 
            print ("ERROR!!!",)
            nErr += 1
        else:
            print ("  OK  ",)
        print (ret)
        del o        

    testOrdered( [], [], [])
    testOrdered( [999], [], [(999,2)])
    testOrdered( [], [999], [(999,1)])
    testOrdered( [0], [0], [(0,0)])

    testOrdered( [-9, 0], [0], [(-9,2), (0,0)])
    testOrdered( [0], [-9, 0], [(-9,1), (0,0)])
    
    testOrdered( [-9, 0, 0], [0, 0], [(-9,2), (0,0), (0,0)])
    testOrdered( [0, 0], [-9, 0, 0], [(-9,1), (0,0), (0,0)])
    
    testOrdered( [-9, 0, 0, 9], [0, 0], [(-9,2), (0,0), (0,0), (9,2)])
    testOrdered( [0, 0], [-9, 0, 0, 9], [(-9,1), (0,0), (0,0), (9,1)])
    
    testOrdered( [-9, 0, 0, 9, 99], [0, 0], [(-9,2), (0,0), (0,0), (9,2), (99,2)])
    testOrdered( [0, 0], [-9, 0, 0, 9, 99], [(-9,1), (0,0), (0,0), (9,1), (99,1)])

    testOrdered( [-9, 0, 0, 9], [0, 0, 99], [(-9,2), (0,0), (0,0), (9,2), (99,1)])
    testOrdered( [0, 0, 99], [-9, 0, 0, 9], [(-9,1), (0,0), (0,0), (9,1), (99,2)])

    print ("-"*60)
    print ("%d ERRORS on %d tests"%(nErr, nTest))
    print ("-"*60)

