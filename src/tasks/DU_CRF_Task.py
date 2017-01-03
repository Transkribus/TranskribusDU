# -*- coding: utf-8 -*-

"""
    CRF DU task core
    
    Copyright Xerox(C) 2016 JL. Meunier

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""
import os
import glob
from optparse import OptionParser

from common.trace import traceln

from crf.FeatureExtractors_PageXml_std import FeatureExtractors_PageXml_StandardOnes
from crf.Model import ModelException
from crf.Model_SSVM_AD3 import Model_SSVM_AD3
from xml_formats.PageXml import MultiPageXml


class DU_CRF_Task:
    """
USAGE:
- Sub-class this class and specialise:
    - ModelClass (class): set the learner+inference methods. As of Dec 2016, only Model_SSVM_AD3 is available. So do not change...
    - sMetadata_Creator (string): will appear in the metadata of the MultiPageXml or PageXml XML files 
    - sMetadata_Comments (string): idem
    - getGraphClass (method): return the Graph class, making sure it defines the labels relevant to the tasks
                                , and optionally the constraints
- create an instance of the sub-class    

See DU_StAZH_b.py

    """
    
    ModelClass = Model_SSVM_AD3
    
    sMetadata_Creator = "XRCE Document Understanding CRF-based - v0.1"
    sMetadata_Comments = None

    def __init__(self, dFeatureConfig={}, dLearnerConfig={}): 
        """
        bPageConstraint: Do we predict under some logical constraints or not?
        """
        self.config_extractor_kwargs = dFeatureConfig
        self.config_learner_kwargs = dLearnerConfig

    def getGraphClass(self):
        raise Exception("Method must be overridden")
    
    #---  WHAT IS BELOW IS GENERIC  -------------------------------
    def getBasicTrnTstRunOptionParser(cls, sys_argv0=None, version=""):
        usage = "%s <model-name> <model-directory> [--trn <col-dir>]+ [--tst <col-dir>]+ [--prd <col-dir>]+"%sys_argv0
        description = """ 
        Train or test the given model or predict using the given model.
        The data is given as a list of DS directories.
        The model is loaded from or saved to the model directory. The model parameters are taken from a Python module named after the model.
        """
    
        #prepare for the parsing of the command line
        parser = OptionParser(usage=usage, version=version)
        
        parser.add_option("-T", "--trn", dest='lTrn',  action="store", type="string"
                          , help="Train or continue previous training session using the given annotated collection.")    
        parser.add_option("-t", "--tst", dest='lTst',  action="store", type="string"
                          , help="Test a model using the given annotated collection.")    
        parser.add_option("-r", "--run", dest='lRun',  action="store", type="string"
                          , help="Run a model on the given non-annotated collection.")    
        return usage, description, parser
    getBasicTrnTstRunOptionParser = classmethod(getBasicTrnTstRunOptionParser)
           
    #----------------------------------------------------------------------------------------------------------    
    def train_test(self, sModelName, sModelDir, lsTrnColDir, lsTstColDir):
        """
        Train a model on the tTRN collections and optionally test it using the TST collections, if not empty
        if bCmpLogit is True, also test using a logit model
        return a test report object
        """
        traceln("-"*50)
        traceln("Training model '%s' in folder '%s'"%(sModelName, sModelDir))
        sConfigFile = os.path.join(sModelDir, sModelName+".py")
        traceln("  Configuration file: %s"%sConfigFile)
        traceln("Train collection(s):", lsTrnColDir)
        traceln("Test  collection(s):", lsTstColDir)
        traceln("-"*50)
        
        #list the train and test files
        #NOTE: we check the presence of a digit before the '.' to eclude the *_du.xml files
        ts_trn, lFilename_trn = self.listMaxTimestampFile(lsTrnColDir, "*[0-9]"+MultiPageXml.sEXT)
        _     , lFilename_tst = self.listMaxTimestampFile(lsTstColDir, "*[0-9]"+MultiPageXml.sEXT)
        
        DU_GraphClass = self.getGraphClass()
        
        traceln("- loading training graphs")
        lGraph_trn = DU_GraphClass.loadGraphs(lFilename_trn, bDetach=True, bLabelled=True, iVerbose=1)
        traceln(" %d graphs loaded"%len(lGraph_trn))
            
        traceln("- creating a %s model"%self.ModelClass)
        mdl = self.ModelClass(sModelName, sModelDir)
        mdl.configureLearner(**self.config_learner_kwargs)
        mdl.saveConfiguration( (self.config_extractor_kwargs, self.config_learner_kwargs) )
        traceln("\t - configuration: ", self.config_learner_kwargs )

        traceln("- retrieving or creating feature extractors...")
        try:
            mdl.loadTransformers(ts_trn)
        except ModelException:
            fe = FeatureExtractors_PageXml_StandardOnes(**self.config_extractor_kwargs)         
            fe.fitTranformers(lGraph_trn)
            fe.clean_transformers()
            mdl.setTranformers(fe.getTransformers())
            mdl.saveTransformers()
        traceln(" done")
        
        traceln("- training model...")
        mdl.train(lGraph_trn, True, ts_trn)
        traceln(" done")
        
        if lFilename_tst:
            traceln("- loading test graphs")
            lGraph_tst = DU_GraphClass.loadGraphs(lFilename_tst, bDetach=True, bLabelled=True, iVerbose=1)
            traceln(" %d graphs loaded"%len(lGraph_tst))
    
            oReport = mdl.test(lGraph_tst)
        else:
            oReport = None, None
            
        return oReport

    def test(self, sModelName, sModelDir, lsTstColDir):
        """
        if bCmpLogit is True, also test using a logit model
        """
        traceln("-"*50)
        traceln("Trained model '%s' in folder '%s'"%(sModelName, sModelDir))
        traceln("Test  collection(s):", lsTstColDir)
        traceln("-"*50)
        
        #list the train and test files
        _     , lFilename_tst = self.listMaxTimestampFile(lsTstColDir, "*[0-9]"+MultiPageXml.sEXT)
        
        traceln("- loading a %s model"%self.ModelClass)
        mdl = self.ModelClass(sModelName, sModelDir)
        mdl.load()
        traceln(" done")
        
        DU_GraphClass = self.getGraphClass()
        
        lPageConstraint = DU_GraphClass.getPageConstraint()
        if lPageConstraint: 
            for dat in lPageConstraint: traceln("\t\t%s"%str(dat))
            
        traceln("- loading test graphs")
        lGraph_tst = DU_GraphClass.loadGraphs(lFilename_tst, bDetach=True, bLabelled=True, iVerbose=1)
        traceln(" %d graphs loaded"%len(lGraph_tst))

        if lPageConstraint:
            lConstraints = [g.instanciatePageConstraints() for g in lGraph_tst]
            oReport = mdl.test(lGraph_tst)
        else:
            oReport = mdl.test(lGraph_tst)
        return oReport

    def predict(self, sModelName, sModelDir, lsColDir):
        """
        Return the list of produced files
        """
        traceln("-"*50)
        traceln("Trained model '%s' in folder '%s'"%(sModelName, sModelDir))
        traceln("Collection(s):", lsColDir)
        traceln("-"*50)
        
        #list the train and test files
        _     , lFilename = self.listMaxTimestampFile(lsColDir, "*[0-9]"+MultiPageXml.sEXT)
        
        traceln("- loading a %s model"%self.ModelClass)
        mdl = self.ModelClass(sModelName, sModelDir)
        mdl.load()
        traceln(" done")
        
        DU_GraphClass = self.getGraphClass()

        lPageConstraint = DU_GraphClass.getPageConstraint()
        if lPageConstraint: 
            for dat in lPageConstraint: traceln("\t\t%s"%str(dat))
        
        traceln("- loading collection as graphs, and processing each in turn. (%d files)"%len(lFilename))
        du_postfix = "_du"+MultiPageXml.sEXT
        lsOutputFilename = []
        for sFilename in lFilename:
            if sFilename.endswith(du_postfix): continue #:)
            [g] = DU_GraphClass.loadGraphs([sFilename], bDetach=False, bLabelled=False, iVerbose=1)
            
            if lPageConstraint:
                traceln("\t- prediction with logical constraints: %s"%sFilename)
                constraints = g.instanciatePageConstraints()
                Y = mdl.predict(g, constraints=constraints)
            else:
                traceln("\t- prediction : %s"%sFilename)
                Y = mdl.predict(g)
                
            doc = g.setDomLabels(Y)
            MultiPageXml.setMetadata(doc, None, self.sMetadata_Creator, self.sMetadata_Comments)
            sDUFilename = sFilename[:-len(MultiPageXml.sEXT)]+du_postfix
            doc.saveFormatFileEnc(sDUFilename, "utf-8", True)  #True to indent the XML
            doc.freeDoc()
            del Y, g
            traceln("\t done")
            lsOutputFilename.append(sDUFilename)
        traceln(" done")

        return lsOutputFilename

    #----------------------------------------------------------------------------------------------------------    
    def listMaxTimestampFile(cls, lsDir, sPattern):
        """
        List the file following the given pattern in the given folders
        return the timestamp of the most recent one
        Return a timestamp and a list of filename
        """
        lFn, ts = [], None 
        for sDir in lsDir:
            lsFilename = sorted(glob.iglob(os.path.join(sDir, sPattern)))  
            lFn.extend(lsFilename)
            if lsFilename: ts = max(ts, max([os.path.getmtime(sFilename) for sFilename in lsFilename]))
        return ts, lFn
    listMaxTimestampFile = classmethod(listMaxTimestampFile)
    
