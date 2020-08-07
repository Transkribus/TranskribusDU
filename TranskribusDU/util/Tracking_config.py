# -*- coding: utf-8 -*-

"""
Configuring the tracking with MLFlow, if MLFlow available

Created on 04/02/2020

Copyright NAVER LABS Europe 2020

@author: JL Meunier
"""

import os


"""
Since a Windows box cannot access the Linux file system, we assume the existence
 of a local MLFlow server
"""

sMLFlowHost = {
                  "nt"      : "127.0.0.1"
                , "posix"   : "anis.int.europe.naverlabs.com"
                , "java"    : None
                }[os.name]

iMLFlowPort = 2020


assert sMLFlowHost, "Internal error: no MLFlow host configured for platform: %s" % os.name
