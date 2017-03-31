
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

def classification_report(confumat, labels=None, target_names=None, digits=2):
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
        labels = np.asarray(len(confumat))
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
    p   = np.diag(confumat)/(eps+confmat.sum(axis=0))
    r   = np.diag(confumat)/(eps+confmat.sum(axis=1))
    f1  = 2*Precision*Recall/(eps+Precision+Recall)
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
    
    
if __name__ == "__main__":
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    y_true = [0, 1, 2, 2, 2]
    y_pred = [0, 0, 2, 2, 1]
    
    confumat = confusion_matrix(y_true, y_pred)
    print confumat
    target_names = ['class 0', 'class 1', 'class 2']
    print classification_report(confumat, target_names=target_names)

    