'''
Created on 30 Nov 2016

Updated 2019-11-20 by JL Meunier
Updated 2020-03-25 by JL Meunier to get the GIT commit as version if possible

@author: meunier
'''

try:
    import git
    sGitCommit = git.Repo(__file__, search_parent_directories=True).head.commit.hexsha
except ImportError:
    sGitCommit = None
    
version= "1.0" if sGitCommit is None else sGitCommit
