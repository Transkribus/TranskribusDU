
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Arnaud Joly <a.joly@ulg.ac.be>
#          Jochen Wersdorfer <jochen@wersdoerfer.de>
#          Lars Buitinck
#          Joel Nothman <joel.nothman@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Jatin Shah <jatindshah@gmail.com>
#          Saurabh Jha <saurabh.jhaa@gmail.com>
#          Bernardo Stein <bernardovstein@gmail.com>
# License: BSD 3 clause

#modified by JLM to use a confusion matrix instead of predicted and GT labels

from functools import reduce

import numpy as np

def confusion_classification_report(confumat, labels=None, target_names=None, digits=3):
    """Build a text report showing the main classification metrics

    Read more in the :ref:`User Guide <classification_report>`.

    Parameters
    ----------
    confumat : confusion matrix

    labels : array, shape = [n_labels]
        Optional list of label indices to include in the report.

    target_names : list of strings
        Optional display names matching the labels (same order).

    digits : int
        Number of digits for formatting output floating point values

    Returns
    -------
    report : string
        Text summary of the precision, recall, F1 score for each class.

    Examples
    --------
    >>> from sklearn.metrics import classification_report
    >>> from sklearn.metrics import confusion_matrix
    >>> y_true = [0, 1, 2, 2, 2]
    >>> y_pred = [0, 0, 2, 2, 1]
    >>> confumat = confusion_matrix(y_true, y_pred)
    >>> target_names = ['class 0', 'class 1', 'class 2']
    >>> print(classification_report(confumat, target_names=target_names))
                 precision    recall  f1-score   support
    <BLANKLINE>
        class 0       0.50      1.00      0.67         1
        class 1       0.00      0.00      0.00         1
        class 2       1.00      0.67      0.80         3
    <BLANKLINE>
    avg / total       0.70      0.60      0.61         5
    <BLANKLINE>

    """

    if labels is None:
        #labels = unique_labels(y_true, y_pred)
        labels = np.asarray(range(len(confumat)))
    else:
        labels = np.asarray(labels)

    last_line_heading = 'avg / total'

    if target_names is None:
        target_names = ['%s' % l for l in labels]
    name_width = max(len(cn) for cn in target_names)
    width = max(name_width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1-score", "support"]
    fmt = '%% %ds' % width  # first column: class name
    fmt += '  '
    fmt += ' '.join(['% 9s' for _ in headers])
    fmt += '\n'

    headers = [""] + headers
    report = fmt % tuple(headers)
    report += '\n'

#     p, r, f1, s = precision_recall_fscore_support(y_true, y_pred,
#                                                   labels=labels,
#                                                   average=None,
#                                                   sample_weight=sample_weight)
    eps=1e-8
    confumat = np.asarray(confumat)
    p   = np.diag(confumat)/(eps+confumat.sum(axis=0))
    r   = np.diag(confumat)/(eps+confumat.sum(axis=1))
    f1  = 2*p*r/(eps+p+r)
    s   = confumat.sum(axis=1)
        
    for i, label in enumerate(labels):
        values = [target_names[i]]
        for v in (p[i], r[i], f1[i]):
            values += ["{0:0.{1}f}".format(v, digits)]
        values += ["{0}".format(s[i])]
        report += fmt % tuple(values)

    report += '\n'

    # compute averages
    values = [last_line_heading]
    for v in (np.average(p, weights=s),
              np.average(r, weights=s),
              np.average(f1, weights=s)):
        values += ["{0:0.{1}f}".format(v, digits)]
    values += ['{0}'.format(np.sum(s))]
    report += fmt % tuple(values)
    return report

def confusion_list_classification_report(lConfumat, labels=None, target_names=None, digits=2):
    """Build a text report showing the P/R/F classification metrics and their stddev

    Parameters
    ----------
    lConfumat : list of confusion matrix of same shape

    labels : array, shape = [n_labels]
        Optional list of label indices to include in the report.

    target_names : list of strings
        Optional display names matching the labels (same order).

    digits : int
        Number of digits for formatting output floating point values

    Returns
    -------
    report : string
        Text summary of the precision, recall, F1 score for each class.

    """

    confumat = reduce(np.add, lConfumat)
    
    if labels is None:
        #labels = unique_labels(y_true, y_pred)
        labels = np.asarray(range(len(confumat)))
    else:
        labels = np.asarray(labels)

    last_line_heading = 'avg / total'

    if target_names is None:
        target_names = list(['%s' % l for l in labels])
    name_width = max(len(cn) for cn in target_names)
    width = max(name_width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1-score", "support"]
    fmt = '%% %ds' % width  # first column: class name
    fmt += '  '
    fmt += ' '.join(['% 9s' for _ in headers])
    fmt += '\n'

    report = "--- MICRO-AVERAGE ---\n"
    headers = [""] + headers
    report += fmt % tuple(headers)
    report += '\n'

#     p, r, f1, s = precision_recall_fscore_support(y_true, y_pred,
#                                                   labels=labels,
#                                                   average=None,
#                                                   sample_weight=sample_weight)
    eps=1e-8
#     confumat = np.asarray(confumat)
    p   = np.diag(confumat)/(eps+confumat.sum(axis=0))
    r   = np.diag(confumat)/(eps+confumat.sum(axis=1))
    f1  = 2*p*r/(eps+p+r)
    s   = confumat.sum(axis=1)
    report += format_as_table(labels, target_names, p, r, f1, s, digits, fmt)

    report += '\n'

    # compute averages
    values = [last_line_heading]
    for v in (np.average(p, weights=s),
              np.average(r, weights=s),
              np.average(f1, weights=s)):
        values += ["{0:0.{1}f}".format(v, digits)]
    values += ['{0}'.format(np.sum(s))]
    report += fmt % tuple(values)
    
    # ---------------------------------------------------------------------------------
    report += "\n--- MACRO-AVERAGE ---\n"
    
    #Now add Variance!
    lP = [np.diag(_confumat)/(eps+_confumat.sum(axis=0)) for _confumat in lConfumat]
    lR = [np.diag(_confumat)/(eps+_confumat.sum(axis=1)) for _confumat in lConfumat]
    lF1 = [2*p*r/(eps+p+r) for p,r in zip(lP, lR)]
    
    avP = average(lP)
    avR = average(lR)
    avF1 = average(lF1)
    report += format_as_table(labels, target_names, avP, avR, avF1, [len(lP)]*len(labels), digits, fmt)
    
    #report += ("       %%%ds\n"%width)%"(stdev)"
    report += "--- (stdev)\n"
    sdevP = stddev( lP ) 
    sdevR = stddev( lR ) 
    sdevF1 = stddev( lF1 )
    report += format_as_table(labels, target_names, sdevP, sdevR, sdevF1, [len(lP)]*len(labels), digits, fmt)

    return report


def format_as_table(labels, target_names, P, R, F, S, digits, fmt):
    report = ""
    for i, _ in enumerate(labels):
        values = [target_names[i]]
        for v in (P[i], R[i], F[i]):
            values += ["{0:0.{1}f}".format(v, digits)]
        values += ["{0}".format(S[i])]
        report += fmt % tuple(values)
    return report

def confusion_accuracy_score(aConfuMat):
    """
    Accuracy score
    """
    eps= 1e-8
    return np.sum(np.diag(aConfuMat)) / (eps+np.sum(aConfuMat)) 

def confusion_PRFAS(cm):
    """
    return the avgP, avgR, avgF1, Accuracy, Support of the confusion matrix
    """
    eps=1e-8
    cm = np.asarray(cm)
    diag = np.diag(cm)
    s   = cm.sum()
    w   = cm.sum(axis=1) / s
    p   = diag/(eps+cm.sum(axis=0))
    r   = diag/(eps+cm.sum(axis=1))
    f1  = 2*p*r/(eps+p+r)
    p,r,f1 = [np.average(_x, weights=w) for _x in (p,r,f1)]
    a   = diag.sum() / (eps+s)
    return p, r, f1, a, s

def test_confusion_PRFAS():
    """
    Line=True class, column=Prediction
TR_B  [[1585  109    4]
TR_I   [ 126 1233   17]
TR_O   [  20   12   82]]

(unweighted) Accuracy score = 90.97 %    trace=2900  sum=3188

             precision    recall  f1-score   support

       TR_B      0.916     0.933     0.924      1698
       TR_I      0.911     0.896     0.903      1376
       TR_O      0.796     0.719     0.756       114

avg / total      0.909     0.910     0.909      3188
    """
    cm = np.array([  [1585,  109,    4]
                      , [ 126, 1233,   17]
                      , [  20,   12,   82]])
    p, r, f1, a, s = confusion_PRFAS(cm)
    def ok(a,b): return abs(a-b) < 0.001
    assert ok(p,0.909), ("p",p)
    assert ok(r,0.910), ("r",r)
    assert ok(f1,0.909), ("f",f1)
    assert ok(a,0.9097), ("a",a)
    assert ok(s,3188), ("s",s)
    
def average(lA):
    return reduce(np.add, lA)/len(lA)

def variance(lA):
    """
    compute the variance elementwise of this list of arrays
    """
    n = len(lA)
    sX = reduce(np.add, lA)
    sX2 = reduce(np.add, [A*A for A in lA])
    
    s2 = (sX2 - sX*sX / n) / (n-1)
    
    return s2

def stddev(lA):
    return np.sqrt(variance(lA))

def evalAdjustedRandScore(lX, lY):
    """
    compute the adjusted Rand index https://en.wikipedia.org/wiki/Rand_index
    
    assumes that objects in cluster are hashable
    """
    # list of all objects
    lOx = [o for X in lX for o in X]
    lOy = [o for Y in lY for o in Y]
    slOx, slOy = set(lOx), set(lOy)
    assert len(slOx) == len(lOx), "multiple occurence of same object in clusters of lX"    # unicity check
    assert len(slOy) == len(lOy), "multiple occurence of same object in clusters of lY"
    assert slOx == slOy         , "clusters lX and lY do not contains the exact same set of objects"

    # cluster index in lX, per object, for all objects
    lCx = [i for i,X in enumerate(lX)   for _o in X] 
    # dictionary object -> cluster index in lY, for all objects
    dCy = {o:i for i,Y in enumerate(lY) for o in Y}

    score = adjusted_rand_score([dCy[o] for o in lOx]    # ref
                              , lCx)
    return score


def test_rand_index():
    
    #     adjusted_rand_score([0, 0, 1, 2], [0, 0, 1, 1])
    #     0.5714285714285715
    import pytest
    
    lref = [ [0, 1], [2], [3] ]
    l    = [ [0, 1], [2, 3] ]

    ref = 0.5714285714285715
    assert abs(evalAdjustedRandScore(l, lref) - ref) < 1e-3
    assert abs(evalAdjustedRandScore(lref, l) - ref) < 1e-3

    # objects should not matter
    lref = [ [0, 11], [2], [333] ]
    l    = [ [0, 11], [2, 333] ]
    assert abs(evalAdjustedRandScore(l, lref) - ref) < 1e-3
    assert abs(evalAdjustedRandScore(lref, l) - ref) < 1e-3

    # order should not matter    
    lref = [ [3], [1, 0], [2] ]
    l    = [ [0, 1], [2, 3] ]
    assert abs(evalAdjustedRandScore(l, lref) - ref) < 1e-3
    assert abs(evalAdjustedRandScore(lref, l) - ref) < 1e-3
   
    # objects and order should not matter    
    lref = [ [3333], [1, 0], [22] ]
    l    = [ [0, 1], [22, 3333] ]
    assert abs(evalAdjustedRandScore(l, lref) - ref) < 1e-3
    assert abs(evalAdjustedRandScore(lref, l) - ref) < 1e-3

    # both must be partition of the same set
    lref = [ [3], [1, 0], [2], [99]]
    l    = [ [0, 1], [2, 3] ]
    with pytest.raises(AssertionError): evalAdjustedRandScore(l, lref)
    with pytest.raises(AssertionError): evalAdjustedRandScore(lref, l)
    l    = [ [0, 1], [2, 3, 2] ]
    with pytest.raises(AssertionError): evalAdjustedRandScore(l, lref)
    with pytest.raises(AssertionError): evalAdjustedRandScore(lref, l)


if __name__ == "__main__":
    from sklearn.metrics import confusion_matrix
    y_true = [0, 1, 2, 2, 2]
    y_pred = [0, 0, 2, 2, 1]
    
    confumat = confusion_matrix(y_true, y_pred)
    target_names = ['class 0', 'class 1', 'class 2']
    print(confusion_classification_report(confumat, target_names=target_names))

    
    
