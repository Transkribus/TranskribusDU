
"""
Component class

Sophie Andrieu, H Dejean, JL Meunier - 2006

Copyright Xerox XRCE 2006

"""
#Adjustement of the PYTHONPATH to include /.../DS/src
import sys, os.path
import logging
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath( sys.argv[0] ) ) ) )

import sys, os.path, types, socket
import config.ds_xml_def as ds_xml
import chrono
import time
import string
import types
import math
import inspect
from collections import defaultdict


import libxml2

### FIX to a problem with libxml2... :-(
def FIX_docSetCompressMode(doc, ratio):
	"""get the compression ratio for a document, ZLIB based """
	#traceln("ratio :", ratio)
	assert ratio in [0,1,2,3,4,5,6,7,8,9], "Internal SW Error zlib in Component.py: ratio=%s"%ratio
	ret = libxml2.libxml2mod.xmlSetDocCompressMode(doc._o, ratio)
	return ret

import XmlConfig
from optparse import OptionParser
from trace import trace, traceln

sHttpHost = "ds.grenoble.xrce.xerox.com"  #server hosting the collections for online browsing

copyright = "Copyright 2005-2016 XEROX"
extXML = ".xml"
extDSXML= '.ds_xml'
extGZXML = ".xml.gz"
extRunFile = ".run"
extHtmFile = ".htm"
extRefFile = ".ref"
configFileBaseName = "config"

""" Tag used to generate documentation with Doxygen """




## The <code>Component</code> class describes a generic component :
# - it manages the command line options with the <code>optparse</code> library
# - it offer a programmatic API to a component
# - it deals with the XML configuration file of a component
# - it offers a unified way to run test cases and to compute a quality measure
#@version 1.5
#@date December 2006
#@author Sophie Andrieu, HD, JLM  - Copyright Xerox XRCE 2006
class Component:

	## The component version
	versionComponent = "1.8"
	
	## The default usage component information
	usageComponent = " [-h] [--version] [--conf=<file>] [--saveconf=<file>] "
	## The usage for <code>-v</code> option (verbose)
	usageVerbose = " [-v] "
	## The usage for <code>--input</code> option
	usageInput = " -i <file>"
	## The usage for <code>--output</code> option
	usageOutput = " -o <file>"
	## The usage for <code>--test</code> option
	usageTest = "[ --test [<dir>|<dir>/xml/<file>] [--incr] [--diff] [--csv] [--out] [--variant=<name>]]"
	
	usageFirstPage = " -f <first-page-number>"
	usageLastPage = " -l <last-page-number>"
	usageEvalOut= "  --evalOut"
	loglevelusage = " --loglevel=(DEBUG|INFO|WARNING|CRITICAL)"
	logDirUsage = " --logDir <dirName>"
	
	_sTestFilePrefix = "test"
	_variantName = "" #we may have several variant of the component that use the same test structure

	## Build a new <code>Component</code> object.
	#@param name The component name
	#@param usageA The usage used to build the <code>OptionParser</code>
	#@param versionA The version used to build the <code>OptionParser</code>
	#@param descriptionA The description used to build the <code>OptionParser</code>
	def __init__(self, name, usageA, versionA, descriptionA):
		
		##Name of the component
		self.name = name 
		
		##usage string
		self.usage = "python %prog" + self.usageComponent + self.usageVerbose + self.usageTest + usageA + self.usageOutput + self.usageEvalOut + self.usageFirstPage + self.usageLastPage + self.loglevelusage + self.logDirUsage
		
		##component version
		self.versionComponent = versionA 
		
		##component description
		self.description = descriptionA
		
		self.initAttributes()

	def setVariant(cls, sVariantName):
		""" Set a different variant name so that we share the same test folder but with non-conflicting filenames.
		"""
		#traceln("- VARIANT: "+sVariantName)
		cls._variantName = sVariantName+"_"
	setVariant = classmethod(setVariant)
		
	##Initialize the attribute values by default
	def initAttributes(self):
		## The input XML data file
		self.inputFileName = "-"
		## The output XML data file
		self.outputFileName = "-"
		
		## Verbose mode
		self.bVerbose = False
		## Debug Mode
		self.bDebug = False
		##Zlib mode
		self.bZLib = False
		self.iZLibRatio = 1  #compression ratio in [0-9], 1 is the smalest ratio, good enough and fastest.

		## logging info level
		self.loggingLevel  = logging.INFO
		self.logDir = "."
		self._bLog = False
		
		## test
		self.bTest = False
		self.bTestIncr = False #incremental mode
		self.bTestCSV = False
		self.bTestOUT = False
		self.sTestVariant = None
		self.sTestCurrentFile = None #the file being processed

		##Internal data: the dictionary of parameters
		self.dParam = None
		##Internal data: the <code>OptionParser</code> instance
		self.parser = None 
		##Internal data: input file descriptor
		self.inputFile = None
		##Internal data: output file descriptor
		self.outputFile = None
		
		self.tag  = ds_xml.sTEXT
		self.evalOutputFile = None
		self.firstPage = 1
		self.lastPage = 99999
		self.listOfPages  = []
		
		self.dicOptionDef = {} # a dictionary of option defintions, used when saving the config file to provide a complete description of each option
		
		#Open Xerox may pass a client identifier, then we must do specific things for accounting
		# See accounting, writeDom
		self.sOXClientID = "" 
		
		
	#--- Command Line Management ------------------------------------------------------
	## Create a command line parser
	# - Initialize the <code>OptionParser</code> with the usage, the version and the description.
	# - Initialize the <code>OptionParser</code> with the default options.
	#@return nothing
	def createCommandLineParser(self):
		self.parser = OptionParser(usage=self.usage, version=self.versionComponent)
		self.parser.description = self.description
		self.add_option("--conf", dest="conf", action="store", type="string", help="get parameters from the XML <file> specified", metavar="<file>")
		self.add_option("--saveconf", dest="saveconf", action="store", type="string", help="save all command line parameters in the specified XML <file>", metavar="<file>")
		self.add_option("-v", "--verbose" , dest="verbose", action="store_true", default=False, help="verbose mode")
		self.add_option("--debug" 	   	  , dest="debug"  , action="store_true", default=False, help="debug mode")
		self.add_option("-z", "--zlib"	  , dest="zlib"   , action="store_true", default=False, help="Compress the output with zlib (See zlib: zcat, zless, gzip, gunzip, etc.). Divide the file zize by ~8.")
		self.add_option("--test", dest="test", action="store", type="string", help="test all files in the given test directory or the given test file only", metavar="<dir>")
		self.add_option("--incr", dest="incr", action="store_true", help="incremental mode, compute only the missing run files", metavar="")
		self.add_option("--diff", dest="diff", action="store_true", help="Show the errors of the test, re-compute the run files only if needed", metavar="")
		self.add_option("--csv", dest="csv", action="store_true", help="Produce a CSV summary file of the test", metavar="")
		self.add_option("--out", dest="testout", action="store_true", help="Store the output of the processing during the test", metavar="")
		self.add_option("--variant", dest="testvariant", action="store", help="Indicate a configuration variant for the test", metavar="")
		self.add_option("-i", "--input", dest="input", default="-", action="store", type="string", help="input XML file", metavar="<file>")
		self.add_option("-o", "--output", dest="output", default="-", action="store", type="string", help="output XML file", metavar="<file>")
		self.add_option("--oxclient", dest="oxclient", action="store", type="string", help="Open Xerox client identifier, for invoicing")
		self.add_option("--logLevel",dest="logLevel", action="store", type="string", help="logging level"   , metavar="<logging Level>")
		self.add_option("--logDir",dest="logDir", action="store", type="string", help="logging Directory"   , metavar="<logging directory>")
		self.add_option("--log", dest="log", action="store_true", default=False, help="log mode")


		#self.add_option("-f", "--first", dest="first", action="store", type="int", default=0, help="first page number", metavar="NN")
		#self.add_option("-l", "--last", dest="last", action="store", type="int", default=10000, help="last page number", metavar="NN")
		#self.add_option("--tag", dest="tag", action="store", default=ds_xml.sTEXT, help="used tag")
		#self.add_option("--evalOut", dest="evalOut", action="store", type="string", default="-", help="eval output")

	
	## Add a parameter to the componenet
	## Syntax is siilar to optparse.OptionParser.add_option (the Python module optparse, class OptionParser, method add_option)
	#@param *args	(passing by position)
	#@param **kwargs (passing by name)
	def add_option(self, *args, **kwargs):
		"""add a new command line option to the parser"""
		self.parser.add_option(*args, **kwargs)
		#keep this info in memory to create weel described config files!
		name = kwargs['dest'] #this will be the firstname of the option
		self.dicOptionDef[name] = (args, kwargs)

	## Parse the command line. 
	#@return a tupple (options, args). The options is a dictionary: <option-name> -> value. The args is a list of string.
	#	<br>For instance "cmd -f 3 toto" will result in the tuple ({'-f':'3'}, 'toto')
	def parseCommandLine(self, app_args=None):
		if app_args:
			(options, args) = self.parser.parse_args(app_args)
		else:
			(options, args) = self.parser.parse_args()
		#we must ignore the options that were not passed
		
		#bullshit!!! options IS NOT a dictionery!! for k,v in dOptions.items():
		# HACK instead (may require adaptation, should optparse change)
		dOptions = {}
		for k,v in options.__dict__.items():
			if v != None: dOptions[k] = v
		return dOptions, args

	#--- Setting the parameters ------------------------------------------------------
	## Set the parameters of the component
	#@param self: the object itself
	#@param dParams a dictionary of destination names (as indicated in add_option) mapping to their values
	#@return nothing
	def setParams(self, dParams):
		if self.bDebug: traceln("Component arguments: ", dParams)
		
		#at this point the dParams maybe the indicated configuration one!
		if dParams.has_key("verbose"): 
			#the value may come from the command line or from the config file or from the programatic dictionary
			v = dParams["verbose"]
			if type(v) == types.BooleanType:
				self.bVerbose = v
			else:
				self.bVerbose = (dParams["verbose"] in ["true", "True"])
			self.setVerbose(self.bVerbose)
		if dParams.has_key("debug"): 
			#the value may come from the command line or from the config file or from the programatic dictionary
			v = dParams["debug"]
			if type(v) == types.BooleanType:
				self.bDebug = v
			else:
				self.bDebug = (dParams["debug"] in ["true", "True"])
		if dParams.has_key("zlib"): 
			#the value may come from the command line or from the config file or from the programatic dictionary
			v = dParams["zlib"]
			if type(v) == types.BooleanType:
				self.bZLib = v
			else:
				self.bZLib = (dParams["zlib"] in ["true", "True"])
		
		if dParams.has_key("oxclient"): self.sOXClientID = dParams["oxclient"]
		
		if dParams.has_key("logLevel"): self.loggingLevel = dParams["logLevel"]
		if dParams.has_key("logDir"): self.logDir = dParams["logDir"]
		if dParams.has_key('log'):self._bLog = dParams['log'] 		
		
		#saveconfig or load config (not both)
		if dParams.has_key("saveconf"): 
			if dParams.has_key("conf"):
				raise Exception("Either save or load a configuration, not both")
			else:
				fnConfig = dParams["saveconf"]
				del dParams["saveconf"]
				self.saveConfig(fnConfig, dParams)
				traceln("SAVING CONFIGURATION AND EXITING.")
				sys.exit(0)
		
		bInput = dParams.has_key("input")  
		if bInput: self.inputFileName  = dParams["input"]
		
		bOutput = dParams.has_key("output")
		if bOutput: self.outputFileName = dParams["output"]

		if dParams.has_key("conf"):
			fnConfig = dParams["conf"]
			for k,v in dParams.items(): del dParams[k]	#empty the ditionary
			self.loadConfig(fnConfig, dParams)  #re-populate the dictionary

		#if some input or output was defined, they take precedence to the config
		if bInput:  dParams["input"]  = self.inputFileName
		if bOutput: dParams["output"] = self.outputFileName

		if dParams.has_key("first"): self.firstPage = int(dParams["first"])
		if dParams.has_key("last"): self.lastPage = int(dParams["last"])
		self.listOfPages = range(self.firstPage,self.lastPage+1)
		if dParams.has_key("evalOut"): self.evalOutputFile = dParams["evalOut"]

		if dParams.has_key("tag"): self.tag = dParams["tag"]  

		self.dParam = dParams #keep in memory

		#TEST mode, ignore any other option, test and return
		if dParams.has_key("test") and dParams["test"]:
			bDiff = dParams.has_key("diff")
			self.bTestIncr = dParams.has_key("incr")
			self.bTestCSV  = dParams.has_key("csv")
			self.bTestOUT  = dParams.has_key("testout")
			if dParams.has_key("testvariant"): self.sTestVariant = dParams["testvariant"]
		
			#if --out:  check if the component has the capability to store the computation output. Typically, old components won't.
			if self.bTestOUT:
				argspec = inspect.getargspec(self.testRun)
				nbargs = len(argspec.args)
				assert nbargs in (2,3), "Internal error: testRun method must have either 2 or 3 arguments"
				if nbargs == 2:
					traceln("WARNING: you requested --out, but this component does not support this feature. Ignoring --out ...")
					self.bTestOUT = False
			self.bTest = True
			self.testDir(dParams["test"], bDiff)
			traceln("Test done, exiting.")
			sys.exit(0)
		elif dParams.has_key("diff") or dParams.has_key("incr") or dParams.has_key("csv") or dParams.has_key("testout") or dParams.has_key("testvariant"):
			traceln("ERROR: use of --diff --incr --csv --out --variant is restricted to the test mode (see --test)")
			raise ComponentException(self.usageTest)
			

	## run the component on the given DOM. May either modify the passed DOM or return a DOM 
	#@doc the input DOM
	#@return the result, usually a DOM, possibly the modified input DOM
	def run(self, doc):
		raise ComponentException("Your component must define a run method!")

	#--- Configuration file ------------------------------------------------------
	##Load a configuration file
	#@param sFileName: the file name
	#@param dParams: OPTIONAL the argument dictionary, by default the internal argument dictionary
	#@return a dictionary of argument to be passed to setParams
	def loadConfig(self, sFileName, dParams={}):
		if self.bVerbose or self.bDebug or self.bTest: traceln("COMPONENT: reading parameters from:", sFileName)
		config = XmlConfig.XmlConfig(self.name, self.versionComponent, self.description, sFileName)
		config.readXmlConfig(dParams)
		return dParams

	##Save the dictionary of arguments into a configuration file
	#@param sFileName: the file name
	#@param dParams: the argument dictionary
	#@return the same dictionary
	def saveConfig(self, sFileName, dParams):
		if self.bVerbose or self.bDebug: traceln("COMPONENT: saving parameters into:", sFileName)
		config = XmlConfig.XmlConfig(self.name, self.versionComponent, self.description, sFileName)
		config.writeXmlConfig(dParams, self.dicOptionDef)
		return dParams
		
	#--- I/O ---------------------------------------------------------------------------
	## get the input file name ("-" denotes stdin)
	#@return the input file name
	def getInputFileName(self):
		"""return the input file name, possibly "-" """
		return self.inputFileName

	## open the input file and return it. Opening mode by default is the read mode.
	##   (the opened file is put in cache and won't be opened twice)
	#@param opening_mode: OPTIONAL the opening mode as for the builtin open function of Python
	#@return the opened input file
	def getInputFile(self, opening_mode="r"):
		"""return an opened input file descriptor"""
		if not self.inputFile:
			if self.inputFileName == "-":
				self.inputFile = sys.stdin
			else:
				self.inputFile = open(self.inputFileName, opening_mode)
		return self.inputFile
	
	##get the output file name ("-" denotes stdout)
	#@return the output file name
	def getOutputFileName(self):
		"""return the output file name, possibly "-" """
		return self.outputFileName
	
	## open the output file and return it. Opening mode by default is the write mode.
	##   (the opened file is put in cache and won't be opened twice)
	#@param opening_mode: OPTIONAL the opening mode as for the builtin open function of Python
	#@return the opened output file
	def getOutputFile(self, opening_mode="w"):
		"""return an opened output file descriptor"""
		if not self.outputFile:
			if self.outputFileName == "-":
				self.outputFile = sys.stdout
			else:
				self.outputFile = open(self.outputFileName, opening_mode)
		return self.outputFile

	## Read a DOM from the input file name of the component or from the given filename
	## The DOM is not cached (multiple calls read multiple times a DOM)
	#@param filename OPTIONAL
	#@return the DOM
	def loadDom(self, filename=None,bGraphic = False):
		if filename == None: 
			filename = self.inputFileName
		libxml2.keepBlanksDefault(False)
		doc =  libxml2.parseFile(filename)
		if bGraphic:
			res = doc.xincludeProcess()
		return doc
	
	## Save the DOM into the output file name
	#@param doc: the DOM
	#@param bIndent OPTIONAL False by default (no indentation of the XML)
	def writeDom(self, doc, bIndent=False):
		if self.bZLib:
			#traceln("ZLIB WRITE")
			try:
				FIX_docSetCompressMode(doc, self.iZLibRatio)
			except Exception, e:
				traceln("WARNING: ZLib error in Component.py: cannot set the libxml2 in compression mode. Was libxml2 compiled with zlib? :", e)
		if bIndent:
			doc.saveFormatFileEnc(self.getOutputFileName(), "UTF-8",bIndent)
		else: 
			#JLM - April 2009 - dump does not support the compressiondoc.dump(self.getOutputFile())
			doc.saveFileEnc(self.getOutputFileName(),"UTF-8")
		
		if self.sOXClientID: self.accounting(doc)
	
	def writeEval(self,doc,f,bIndent=False):
		if doc == None: return None
		fout = None
		try:
			fout = open(f,"w")
		except: 
			traceln("impossible to open %s" %f)
		if fout:
			fout.write(doc)	
		return None  
	
	#Open Xerox accounting: we create a .cost file, containing the client ID and the number of pages
	# space separation
	def accounting(self, doc):
		ctxt = doc.xpathNewContext()
		nPageTot = int(ctxt.xpathEval('count(//PAGE)'))
		ctxt.xpathFreeContext()
		
		#Cost "statement"  :-)
		#sCost = time.strftime("%Y/%m/%d\t%H:%M:%S\tGMT", time.gmtime()) #%Z can be verbose, e.g. Romance Standard Time or empty
		sCost = self.name.expandtabs(1).strip()
		sCost = sCost + "\t%d" % nPageTot 
		sCost = sCost + "\t%s" % self.sOXClientID.expandtabs(1).strip()
		sCost = sCost + time.strftime("\t%Y/%m/%d\t%H:%M:%S", time.localtime()) #%Z can be verbose, e.g. Romance Standard Time or empty
		#Store the cost file
		sCostFile = self.getOutputFileName()+".cost"
		fd = open(sCostFile, "w")
		fd.write(sCost)
		fd.write("\n")
		fd.close()
		traceln("COST STATEMENT :-) : ", sCost)
		
	#----------------------------------------------------------------------------
	# TEST MODE
	
	## Manage the test execution on all files included in the test directory.
	# For each file included in the directory:
	#	 - it calls the <code>testRun(file)</code> method which is specialized by each component.
	#	 - if a reference result exists (in a .ref file), it calls <code>testCompare</code>
	#	 - then the individual results are aggragated using the testInit, testRecord, testReport methods, which can
	#		   be specialized by the component if a standard precision/recall measure does not fit.
	#@param self this object
	#@param dir The directory which include the XML data files to test
	#@param bDiff boolean, if True, we perform a diff not a test
	def testDir(self, dir, bDiff=False):
		nbMinus = 60 #1/2 width of the horizontal lines of '-'

		if self.bDebug or self.bVerbose:
			traceln("-"*nbMinus)
			if bDiff:
				traceln("--- DIFF MODE ---")
			else:
				traceln("--- TEST MODE ---")

		self.testInit()
		
		if not os.path.exists(dir):
			traceln("*** NO SUCH FILE OR DIRECTORY: ", dir)
			if bDiff:
				traceln("EXITING")
				sys.exit(0)
			traceln("*** DO YOU WANT TO CREATE A NEW TESTSET?? [y/n]")
			s = raw_input()
			if s.lower() not in ["yes", "y"]:
				traceln("*** EXITING...")
				sys.exit(0)
			#ok, create this directory. testDir will make the rest of the structure
			os.mkdir(dir)
			bCREATION_MODE = True
		else:
			bCREATION_MODE = False

		#do we run a variant?
		if self.sTestVariant != None:
			traceln("- variant = "+self.sTestVariant) 
			self.setVariant(self.sTestVariant)
					
		#in fact we may receive a directory or only one of the XML file in a test structure
		bSingleFile = False
		if os.path.isdir(dir):
			#ok we got a directory
			self.testDirInit(dir)  #raise a ComponentException if the structure is not valid
			lTestFile = []
			for fn in os.listdir(self.testDirXML):
				if fn.endswith(extXML) or fn.endswith(extDSXML) or fn.endswith(extGZXML):
					lTestFile.append(fn)
				else:
					traceln("--- WARNING: skipping file (not XML?): %s"%fn)
			lTestFile.sort()
		else:
			#ok we must have got something like "TEST1/xml/toto.xml"
			if not os.path.isfile(dir): raise ComponentException("%s is not a valid file or directory"%dir)
			dir, testFile = os.path.split(dir)
			dir, subdir = os.path.split(dir)
			if subdir != "xml": raise ComponentException("%s is not a valid test file in a xml directory"%dir)
			lTestFile = [ testFile ]
			bSingleFile = True
			#ok, now we got a supposedely test structure
			self.testDirInit(dir)   #raise a ComponentException if the structure is not valid
			
			
		#remove any .run file before a test (and new in v1.8) any output
		if not bDiff and not self.bTestIncr:
			for fn in lTestFile:
				fnrun = self.getRunFileName(fn)
				if os.path.exists(fnrun): os.remove(fnrun) 
				if self.testDirOUT: #otherwise it means the out folder does not even exist (so, an old component created the test directory structure)
					fnout = self.getOutFileName(fn)
					if os.path.exists(fnout): os.remove(fnout) 
		
		if bCREATION_MODE:
			traceln("*** Now create the test configuration file using the   --saveconf %s   option"%self.testConfigFile)
			traceln("*** Then populate the %s directory with XML samples." % self.testDirXML)
			traceln("*** EXITING...")
			sys.exit(0)
				
		#load the configuration
		dParams = self.loadConfig(self.testConfigFile)
		self.setParams(dParams)
		
		#test each test file
#		self.beginChrono()  #PB: the stack of chrono is often incorrect => we get a wrong duration...
		cTot = chrono.Chrono().on()
		lMissingRefFile = []
		lMissingRunFile = []  #for the diff mode
		nbRun = 0
		nbCmp = 0
		ii = 0
		for fn in lTestFile:
			ii += 1
			traceln("-"*nbMinus)
			self.sTestCurrentFile = fn
			fnXML = os.path.join(self.testDirXML, fn)
			fnRun = self.getRunFileName(fn)
			fnRef = self.getRefFileName(fn)
			
			#now either create the run or in diff mode create it only if needed
			traceln("--- %4d / %d"%(ii,len(lTestFile)))
			bReusePreviousRun = False
#			if (bDiff or self.bTestIncr) and os.path.isfile(fnRun):
			if bDiff or self.bTestIncr:
				#we re-compute only if: 1) the run is missing or 2) the run is outdated
				mtimeXML = os.path.getmtime(fnXML)
				try:
					mtimerun = os.path.getmtime(fnRun)
					if mtimerun >= mtimeXML: 
						bReusePreviousRun = True
					else:
						traceln(" - obsolete run file: %s"% fnRun)
				except os.error:
					traceln(" - missing run file: %s"% fnRun)
					pass
			if bReusePreviousRun:
				#do not re-compute it
#				try:
					f = open(fnRun, "r"); rundata = f.read(); f.close()
#				except IOError, e:
#					traceln("WARNING: SKIPPING FILE %s: %s"%(fn, e))
#					lMissingRunFile.append(fn)
#					continue
			else:
				traceln("--- Running on : %s"%fn)
				if self.bTestOUT: 
					fnOut = self.getOutFileName(fn)
					self.outputFileName = fnOut
					rundata = self.testRun( fnXML, fnOut ) #RUN on THIS ONE! keep the output in the given 2nd file
					self.outputFileName = None
				else:
					rundata = self.testRun( fnXML ) #RUN on THIS ONE!
				#store this new result in a run file
				f = open(fnRun, "w"); f.write(rundata); f.close()
				nbRun += 1
			
			#read the ref data
			if os.path.exists(fnRef):
				f = open(fnRef, "r"); refdata = f.read(); f.close()
				if bDiff: traceln("--- Diffing for: %s"%fn)
				#Compare both results
				cmpData = self.testCompare(refdata, rundata, bDiff)
				nbCmp += 1
			
				#Record this compareason for the report
				self.testRecord(fn, cmpData)
			else:
				traceln("*** no reference data for %s"%fn, fnRef)
				lMissingRefFile.append(fn)
#		duration = self.endChrono()
		duration = cTot.off()
		
		#Ok, NOW REPORT!
		"""
		here we construct a typical test report, including time information etc
		"""
		sLineLn = "-"*min(int(nbMinus*1.5), 79) + "\n"
		sRpt = sLineLn
		sRpt += "  --- TEST REPORT - %s ---\n"%time.ctime()
		sRpt += sLineLn
		sRpt += " - OS (socket.gethostname())= %s\n"%str(socket.gethostname())
		sRpt += " - Python (sys.executable) = %s\n"%str(sys.executable)
		sRpt += " - Python (sys.version_info)= %s\n"%str(sys.version_info)
		sRpt += " - Component = %s \n"%os.path.abspath(sys.argv[0])
		sRpt += " - Component version = %s \n"%self.getVersion()
		sRpt += " - Component variant: %s\n"%self.sTestVariant
		sRpt += " - Command = %s %s \n"% (sys.executable , " ".join(sys.argv))
		sRpt += " - Params =\n"
		dParam = self.getParams()
		lk = dParam.keys()
		lk.sort()
		for k in lk:
			sRpt += " "*11 + "%s = %s\n" % (k, `dParam[k]`)
		sRpt += sLineLn
		sRpt += "\n--- TEST REPORT ---\n"
		sRpt += "\n"
		sRpt += self.testReport()
		sRpt += sLineLn
		
		sMsg = "\n[%.1fs] total time   %d files" % (duration, len(lTestFile))
		if lTestFile:
			sMsg += "   (%.1f second/file)" % (duration / len(lTestFile))
		#sMsg += "   (%d runs and %d run-ref comparisons)"%(nbRun, nbCmp)
		sRpt += "%s\n" %sMsg
		
		if lMissingRefFile: 
			sRpt += "\n"
			sRpt += "*** WARNING ***: %d missing reference files out of %d XML sample files\n"%(len(lMissingRefFile), len(lTestFile))
			sRpt += "	   Missing for: %s\n"%lMissingRefFile
		if lMissingRunFile: 
			sRpt += "\n"
			sRpt += "*** WARNING ***: %d missing run files out of %d XML sample files\n"%(len(lMissingRunFile), len(lTestFile))
			sRpt += "	   Missing for: %s\n"%lMissingRunFile
		sRpt += "\n"
			
		#show it
		traceln(sRpt)
		
		#store it (not when one single file")
		if not(bSingleFile):
			sBaseReportFile = self.getReportFileName(self.testFolder, self._sTestFilePrefix)
			sReportFile = sBaseReportFile + ".txt"
			trace("\tstoring the report in ", sReportFile, "   ...")
			f = open(sReportFile, 'w');	f.write(sRpt); f.close()
			traceln(" done.")

			#also make an HTML report
			sHtmlReportFile = sBaseReportFile + ".htm"
			sHtmlRpt = self.makeHTMLReport()
			trace("\tstoring the report in ", sHtmlReportFile, "   ...")
			f = open(sHtmlReportFile, 'w');	f.write(sHtmlRpt); f.close()
			traceln(" done.\n")			
			
			if self.bTestCSV:
				sCSV = self.makeCSVReport()
				sCSVReportFile = sBaseReportFile + ".csv"
				trace("\tstoring the CSV report in ", sCSVReportFile, "   ...")
				f = open(sCSVReportFile, 'w');	f.write(sCSV); f.close()
				traceln(" done.\n")
				
		
		
		
		
	## Initialize directories for the test option :
	# - The <testDir>/xml directory stores XML data files which will be read
	# - The <testDir>/run directory is used to store RUN result files
	# - The <testDir>/ref directory is used to store REF result files
	# - the directory contains a config.xml file
	# EITHER none of the above directories exit, and they are created
	# OR ALL of them exist
	# OTHERWISE, raise an Exception
	# @param dir The test directory
	def testDirInit(self, dir):
		self.testFolder    = dir
		self.testDirXML = os.path.join(dir, "xml")
		self.testDirRUN = os.path.join(dir, "run")
		self.testDirREF = os.path.join(dir, "ref")
		self.testDirOUT = os.path.join(dir, "out")	#only used if the --out modifier has been specified to keep the output
		self.testConfigFile = os.path.join(dir, self._variantName+configFileBaseName+extXML)
		traceln("- config file is: ", self.testConfigFile)


	#@optional param filename: where to store the result of the process (not the result of the test)
	# Note: for ascendent compatibility, we inspect self to determine if the optional parameter is supported or not.
	# (old component do not have this capability)
	

							
		if	 os.path.isdir(self.testDirXML)  or os.path.isdir(self.testDirRUN)  or os.path.isdir(self.testDirREF):
			if os.path.isdir(self.testDirXML) and os.path.isdir(self.testDirRUN) and os.path.isdir(self.testDirREF):
				if	 not os.path.isfile(self.testConfigFile):
					raise ComponentException("Config file missing: %s"%self.testConfigFile)
			else:
				raise ComponentException("Invalid test structure, needs the 'xml', 'run', 'ref' directories!")
		else:
			#ok, let's create everything
			os.mkdir(self.testDirXML)
			os.mkdir(self.testDirRUN)
			os.mkdir(self.testDirREF)
			os.mkdir(self.testDirOUT)
		
		if self.bTestOUT:
			if not(os.path.isdir(self.testDirOUT)): os.mkdir(self.testDirOUT)	#create the out directory, because old components were not creating it, so the test directory structure can be incomplete
		if not(os.path.isdir(self.testDirOUT)): self.testDirOUT = "" #we have never been in --out mode, so no need to take care of unexisting output
			
			
	#specific basename function
	def getBasename(self, fileName):
		if fileName.endswith(".ds_xml"):
			i = -7  
		elif fileName.endswith(".gz"): 
			i = -7 	#  .xml.gz
		else:
			i = -4 	#  .ref  .run   .xml
		assert fileName[i] == ".", "internal error, unexpected filename: "+fileName
		return os.path.basename(fileName)[:i]
		
	## Build the REF path file name
	#@param fileName The input XML file name
	def getRefFileName(self, fileName):
		return self.testDirREF + os.sep + self.getBasename(fileName) + extRefFile
	
	## Build the RUN path file name
	#@param fileName The input XML file name
	def getRunFileName(self, fileName):
		return self.testDirRUN + os.sep + self._variantName + self.getBasename(fileName) + extRunFile

	## Build the HTML RUN path file name
	#@param fileName The input XML file name
	def getHtmRunFileName(self, fileName):
		return self.testDirRUN + os.sep + self._variantName + self.getBasename(fileName) + extHtmFile	

	def getReportFileName(self, testFolder, testFilePrefix):
		return os.path.normpath(testFolder + os.sep + self._variantName + testFilePrefix + time.strftime("_%Y%m%d_%H_%M_%S", time.localtime())) 
	 
	## Build the OUT path file name
	#@param fileName The input XML file name
	def getOutFileName(self, fileName):
		#note: we do not remove the .gz because we want to preserve the naming conventions of the user
		return self.testDirOUT + os.sep + self._variantName + os.path.basename(fileName)
			
	## Manage the test execution on a specified file
	#@param filename The file to test
	#@optional param filename: where to store the result of the process (not the result of the test)
	# Note: for ascendent compatibility, we inspect self to determine if the optional parameter is supported or not.
	# (old component do not have this capability)
	#@return a string, understandable to a human and to the <code>testCompare</code> method
	def testRun(self, file, outFile=None):
		raise ComponentException("SOFTWARE ERROR: your component must define a testRun method")
	
	## Compare two Python strings produced by testRun
	#@param refdata the reference data (a black box data meaningful to the tested component)
	#@param rundata the recent	data (a black box data meaningful to the tested component)
	#@param bVisual OPTIONAL, False by default, if True, display more information regarding the differences
	#@return a tuple (<number-of-ok>, <number-of-error>, <number-of-misses>) or something else, but in this case
	#   the component must define its own <code>testInit</code>, <code>testRecord</code> and <code>testReport</code> methods.
	def testCompare(self, refdata, rundata, bVisual=False):
		raise ComponentException("SOFTWARE ERROR: your component must define a testCompare method")
	
	def testCompare_InfoExtraction(self, refdoc, rundoc, sXpathExpr, funCompare=None, funNormalize=None, bVisual=False):
		"""
		A generic testCompare, when we search for one particular extracted information per page.
		
		It expects the same XML structure in refdoc and rundoc  (a DOM each or a serialized XML)
		
		The given xpath expression must select one value in each XML for each page
		
		Optionally, each value is normalized using the funNormalize function, if given
		
		The values are compared using the given compare function, which return a true or false value, or using the standard Python comparison if funCompare is None
		
		return the usual (nok, nerr, nmiss, ltisRefsRunbErrbMiss)
		 
		"""
		nok, nerr, nmiss = 0,0,0
		
		if isinstance(refdoc, str): refdoc = libxml2.parseMemory(refdoc, len(refdoc))
		if isinstance(rundoc, str): rundoc = libxml2.parseMemory(rundoc, len(rundoc))
		
		refctxt, runctxt = refdoc.xpathNewContext(), rundoc.xpathNewContext()
		refctxt.setContextNode(refdoc.getRootElement())
		runctxt.setContextNode(rundoc.getRootElement())
		
		reflpnum, runlpnum = refctxt.xpathEval(sXpathExpr), runctxt.xpathEval(sXpathExpr)
		
		itreflLen, itrunlLen = iter(reflpnum), iter(runlpnum)

		i = 0
		ltisRefsRunbErrbMiss = list()
		try:
			while True:
				i += 1
				bErr, bMiss = False, False
				ref, run = itreflLen.next().getContent(), itrunlLen.next().getContent()
				if funNormalize:
					srunnorm = funNormalize(run) #using also our normalization in addition to the standard one				
					srefnorm = funNormalize(ref)
				else:
					srunnorm, srefnorm = run, ref
				#traceln((i, ref, run))
				if run: #it found something
					if funCompare:
						bOk = funCompare(srunnorm, srefnorm) 
					else:
						bOk = (srunnorm == srefnorm)
					if bOk:
						nok += 1
						if bVisual: traceln("<OK> *** : page %d: '%s' got '%s'"%(i, ref, run))
					else:
						nerr += 1   #something wrong
						bErr = True
						if ref: 
							nmiss += 1 #but this is also a miss
							bMiss = True
						if bVisual: traceln("<ERR>	: page %d: '%s' expected but got *** '%s'"%(i, ref, run))
				else:   #it found nothing
					if ref:
						nmiss += 1  #it missed something	
						bMiss = True
						if bVisual: traceln("<MISS>   : page %d: '%s' expected ***"%(i, ref))
					else:
						nok += 1  #just fine!!
				ltisRefsRunbErrbMiss.append( (i, ref, run, bErr, bMiss) )
						
		except StopIteration:
			pass
		refctxt.xpathFreeContext(), runctxt.xpathFreeContext()

		assert len(reflpnum) == len(runlpnum), "***** ERROR: inconsistent ref (%d) and run (%d) lengths. *****"%(len(reflpnum), len(runlpnum))

		return nok, nerr, nmiss, ltisRefsRunbErrbMiss		

	##called at the begin of the test to initialize internal data
	def testInit(self):
		#self.ltFileOkErrMiss = []
		#in case we deal with the results of multiple tasks in one report
		self.dic_ltFileOkErrMiss = defaultdict(list)  
		
	##Predefined behaviour for when the <code>tesrRun</code> returns the number of Ok, Wronf, Missed items. This method records them
	#@param a tuple (nOk, nErr, nMiss)
	def testRecord(self, filename, cmpData):
		
		
		if isinstance(cmpData, dict):
			#new way of returning the results for multiple tasks!!
			ltTaskName_cmpData = cmpData.items()
			ltTaskName_cmpData.sort()
			ltOkErrMissltis = list()
			for taskName, cmpData in ltTaskName_cmpData:
				(nOk, nErr, nMiss, ltisRefsRunbErrbMiss) = cmpData
				
				self.dic_ltFileOkErrMiss[taskName].append( (filename, nOk, nErr, nMiss, ltisRefsRunbErrbMiss) )
				
				ltOkErrMissltis.append( (taskName, nOk, nErr, nMiss, ltisRefsRunbErrbMiss) )

				fP, fR, fF = self.computePRF(nOk, nErr, nMiss)
				sRpt = "%20s  Doc %d: %s\t%6s\t%6s\t%6s\t%s"%(taskName, len(ltOkErrMissltis), self.formatPRF(fP, fR, fF), nOk, nErr, nMiss, filename)
				traceln(sRpt)				
			traceln()	
			self.testRecordHtml(filename, ltOkErrMissltis, None, None, None)
				
			
			return
		
		assert not( isinstance(cmpData, dict) ), "INTERNAL ERROR"
		
		#usual code - UNCHANGED
		if len(cmpData) == 3: #old style, no HTML report can be  produced
			(nOk, nErr, nMiss) = cmpData
			errorMsg = "either testRun returns a tuple (<int>, <int>, int>) or the component defines its own testInit, testRecord and testReport methods"
			assert type(nOk)   == types.IntType, errorMsg
			assert type(nErr)  == types.IntType, errorMsg
			assert type(nMiss) == types.IntType, errorMsg
			ltisRefsRunbErrbMiss = None
		else:
			assert len(cmpData) == 4, "either testRun returns a tuple (<int>, <int>, int>, list-of-item-test) or the component defines its own testInit, testRecord and testReport methods"
			(nOk, nErr, nMiss, ltisRefsRunbErrbMiss) = cmpData
		
		#self.ltFileOkErrMiss.append( (filename, nOk, nErr, nMiss) )
		self.dic_ltFileOkErrMiss[None].append( (filename, nOk, nErr, nMiss, None) )

		#some nice trace
		fP, fR, fF = self.computePRF(nOk, nErr, nMiss)
# 		sRpt = "Doc %d: %s\t%6s\t%6s\t%6s\t%s\n"%(len(self.ltFileOkErrMiss), self.formatPRF(fP, fR, fF), nOk, nErr, nMiss, filename)
		sRpt = "Doc %d: %s\t%6s\t%6s\t%6s\t%s\n"%(len(self.dic_ltFileOkErrMiss[None]), self.formatPRF(fP, fR, fF), nOk, nErr, nMiss, filename)
		traceln(sRpt)
		
		if ltisRefsRunbErrbMiss != None:
			self.testRecordHtml(filename, ltisRefsRunbErrbMiss, nOk, nErr, nMiss)
			
		return
			
	##Report about the test that has been done, in mode nOk, nErr, nMiss, display it on stderr by default
	#@return the report
	def testReport(self):
# 		return self.makeReport(self.ltFileOkErrMiss)
# 
# 	def makeReport(self, ltFileOkErrMiss):
		"""
		build a nice textual report given the list of tuple (filename, nok, nerr, nmiss) 
		"""
		sRpt = ""
		
		lItems = self.dic_ltFileOkErrMiss.items()
		lItems.sort()
		for taskName, lt_Filename_nOk_nErr_nMiss_ltisRefsRunbErrbMiss in lItems:
			
			
			lPRF = []
			if taskName != None:
				sRpt += "\n\n ***** TASK = %s *****\n"%taskName
			#sRpt += " Doc Prec. Recall F1\t   nOk\t  nErr\t nMiss\tFilename\n"
			sRpt += " Doc %7s%7s%7s%6s%6s%6s  \tFilename\n" %("Prec", "Recall", "F1", "Ok", "Err", "Miss")
			docid = 0
			for filename, nOk, nErr, nMiss, _useless in lt_Filename_nOk_nErr_nMiss_ltisRefsRunbErrbMiss:
				docid += 1
				fP, fR, fF = self.computePRF(nOk, nErr, nMiss)
				lPRF.append( (fP, fR, fF) )
				sRpt += "%4d %s%6s%6s%6s \t%s\n"%(docid, self.formatPRF(fP, fR, fF), nOk, nErr, nMiss, filename)
	
			#MICRO
			sumNOk	 = sum( [rec[1] for rec in lt_Filename_nOk_nErr_nMiss_ltisRefsRunbErrbMiss] )
			sumNErr  = sum( [rec[2] for rec in lt_Filename_nOk_nErr_nMiss_ltisRefsRunbErrbMiss] )
			sumNMiss = sum( [rec[3] for rec in lt_Filename_nOk_nErr_nMiss_ltisRefsRunbErrbMiss] )
			fP, fR, fF = self.computePRF(sumNOk, sumNErr, sumNMiss)
			sRpt += "\n\n"
	#		sRpt += "- micro-average\n"
	#		sRpt += "%s\t%6s\t%6s\t%6s\n"%(self.formatPRF(fP, fR, fF), sumNOk, sumNErr, sumNMiss)
	#		sRpt += "%3s %s%6s%6s%6s \tTOTAL\n"%("", self.formatPRF(fP, fR, fF), sumNOk, sumNErr, sumNMiss)
			if taskName != None:
				sRpt += " All %s%6s%6s%6s \t%s\n"%(self.formatPRF(fP, fR, fF), sumNOk, sumNErr, sumNMiss, taskName)
			else:
				sRpt += " All %s%6s%6s%6s\n"%(self.formatPRF(fP, fR, fF), sumNOk, sumNErr, sumNMiss)
			
			#MACRO
			ma_cnt, ma_avg_prec	, ma_sdv_prec   = self.av_stddev_f(map(lambda x:x[0], lPRF))
			ma_cnt, ma_avg_recall  , ma_sdv_recall = self.av_stddev_f(map(lambda x:x[1], lPRF))
			ma_cnt, ma_avg_F1	  , ma_sdv_F1	 = self.av_stddev_f(map(lambda x:x[2], lPRF))
			sRpt += "\n"
	#		sRpt += "- macro-average:\n"
	#		sRpt += "%s\n"%(self.formatPRF(ma_avg_prec, ma_avg_recall, ma_avg_F1))
			sRpt += "%s\t- MACRO-AVERAGE\n"%(self.formatPRF(ma_avg_prec, ma_avg_recall, ma_avg_F1))
			sRpt += "%s\t (standard deviation)\n"%(self.formatPRF(ma_sdv_prec, ma_sdv_recall, ma_sdv_F1, True))

		return sRpt

	def makeCSVReport(self):
		"""
		build a CSV report given the list of tuple (filename, nok, nerr, nmiss) 
		"""
		sRpt = ""
		line = 0
		#sRpt += " Doc Prec. Recall F1\t   nOk\t  nErr\t nMiss\tFilename\n"
		lItems = self.dic_ltFileOkErrMiss.items()
		lItems.sort()		
		for taskName, lt_Filename_nOk_nErr_nMiss_ltisRefsRunbErrbMiss in lItems:
			sRpt += "\n\n%s\n"%taskName
			sRpt += "Doc,Precision,Recall,F1,#ok,#error,#miss,filename\n"
			line += 4
			prevline = line
			docid = 0
			for filename, nOk, nErr, nMiss, _useless in lt_Filename_nOk_nErr_nMiss_ltisRefsRunbErrbMiss:
				docid += 1
				fP, fR, fF = self.computePRF(nOk, nErr, nMiss)
				sRpt += "%d,%s,%s,%s,%s,%s,%s,%s\n"%(docid
											, Component.formatFloatPercent(fP), Component.formatFloatPercent(fR), Component.formatFloatPercent(fF)
											, nOk, nErr, nMiss, filename)
				line += 1
	
			#MICRO
	#		sumNOk	 = sum( [rec[1] for rec in ltFileOkErrMiss] )
	#		sumNErr  = sum( [rec[2] for rec in ltFileOkErrMiss] )
	#		sumNMiss = sum( [rec[3] for rec in ltFileOkErrMiss] )
	#		fP, fR, fF = self.computePRF(sumNOk, sumNErr, sumNMiss)
	#		sRpt += ",%s,%s,%s,%s,%s,%s,micro\n"%(Component.formatFloatPercent(fP), Component.formatFloatPercent(fR), Component.formatFloatPercent(fF)
	#									, sumNOk, sumNErr, sumNMiss)
			#Let the spreadsheet compute all this
			n=line+1
			sRpt += ",=E%d/(E%d+F%d)"%(n,n,n) #precision
			sRpt += ",=E%d/(E%d+G%d)"%(n,n,n) #recall
			sRpt += ",=2*B%d*C%d/(B%d+C%d)"%(n,n,n,n) #F1
			
			sRpt += ",=sum(E%d:E%d)"%(prevline, line) #sum of ok
			sRpt += ",=sum(F%d:F%d)"%(prevline, line) #sum of err
			sRpt += ",=sum(G%d:G%d)"%(prevline, line) #sum of miss
			sRpt += ",micro\n"
			line += 1

		return sRpt

	def makeHTMLReportHeader(self, sBase, sTarget, sCSS, sTitle, sH1):
		sBaseTag = ""
		if sBase or sTarget: 
			sBaseTag = '<base'
			if sBase: sBaseTag += ' href="%s"'%sBase
			if sTarget: sBaseTag += ' target="%s"'%sTarget
			sBaseTag += '/>'
		sRpt = """<?xml version="1.0" encoding="UTF-8" ?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
	<title>%s</title>
	%s
	%s
</head>
<body>
""" % (sTitle, sBaseTag, sCSS)
		sRpt += "<h1>%s</h1>"%sH1
		sRpt += "<p>%s <i>'%s'</i> version=%s</p>"%(self.__class__, self.name, self.versionComponent)
		if self._variantName: sRpt += "<b>variant = %s</b> &nbsp; "%self._variantName[:-1]
		s = self._variantName+configFileBaseName+extXML
		if os.path.exists(s):
			sRpt += '<a href="%s">%s</a>'%(s, s)
		else:
			sRpt += s 
		sRpt += "<p/>"		
		return sRpt
	
	def makeHTMLReport(self):
		"""
		build a CSV report given the list of tuple (filename, nok, nerr, nmiss) 
		"""
		sCollecDir = os.path.dirname(self.testDirXML)
		sCollec = os.path.basename(sCollecDir)			
		sRpt = self.makeHTMLReportHeader(None, None, "", sCollec, sCollec)

		lItems = self.dic_ltFileOkErrMiss.items()
		lItems.sort()
		for taskName, lt_Filename_nOk_nErr_nMiss_ltisRefsRunbErrbMiss in lItems:
			sRpt += self.genHtmlTableReport(sCollec, taskName, lt_Filename_nOk_nErr_nMiss_ltisRefsRunbErrbMiss)
		sRpt += """
</body>
</html>
"""
		return sRpt

	def genHtmlTableReport(self, sCollec, taskName, lt_Filename_nOk_nErr_nMiss_ltisRefsRunbErrbMiss):
	
		sRpt = """
	<table width="100%">
		<tr align="left">
			<th>Doc</th>
			<th>Precision</th>
			<th>Recall</th>
			<th>F1</th>
			<th>ok</th>
			<th>err</th>
			<th>miss</th>
			<th>filename</th>
		</tr>
"""

		if taskName != None: sRpt = "<hr/><h2>%s</h2>\n"%taskName + sRpt
		docid = 0
		for filename, nOk, nErr, nMiss, _useless in lt_Filename_nOk_nErr_nMiss_ltisRefsRunbErrbMiss:
			docid += 1
			fP, fR, fF = self.computePRF(nOk, nErr, nMiss)
			sRpt += "<tr>"
			for s in  (docid
							, Component.formatFloatPercent(fP), Component.formatFloatPercent(fR), Component.formatFloatPercent(fF)
							, nOk, nErr, nMiss):
				sRpt += "<td>%s</td>\n" % s
			sRpt += """<td><a href="http://%s/v/%s/%s" target="dla_pdf">%s</a>\n""" % (sHttpHost, sCollec, self.getBasename(filename), filename)
			if os.path.exists(self.getHtmRunFileName(filename)):
				sRpt += """ &nbsp; <a href="../%s">diff</a>""" % (self.getHtmRunFileName(filename))
			sRpt += """ &nbsp; <a href="http://%s/x/%s/%s">xml</a>""" % (sHttpHost, sCollec, self.getBasename(filename))
			sRpt += """</td>""" 
			sRpt += "<tr>\n"
		sRpt += "</table>"
		return sRpt

	def testRecordHtml(self, filename, data, nOk, nErr, nMiss):
		
		
		if nOk == None:
			assert nErr == None and nMiss == None, "INTERNAL ERROR"
			#we are reporting on multiple tasks!!
			lltisRefsRunbErrbMiss = data #this is a list of (taskName, nOk, nErr, nMiss, ltisRefsRunbErrbMiss)
		else:
			lltisRefsRunbErrbMiss = [ (None, nOk, nErr, nMiss, data) ]
			
		#let's produce an HTML report!! 
		sCollecDir = os.path.dirname(self.testDirXML)
		sCollec = os.path.basename(sCollecDir)
		sFile =   os.path.basename(self.getRefFileName(filename))[:-4]
		sViewBaseUrl = "http://" + sHttpHost 
		
		
		fHtml = open(self.getHtmRunFileName(filename), "w")
		
		sCss = """
<style type="text/css">
.OK {
color: green;
}
.Error {
color: red;
}
.Error\+Miss {
color: darkred;
}
.Miss {
color: orange;
}
</style>	   
"""
		sRpt = self.makeHTMLReportHeader(sViewBaseUrl, "dla_pdf", sCss
										 , sCollec + " - " + sFile
										 , sCollec + " - " + sFile)

		fHtml.write(sRpt)
		
		#sRpt += " Doc Prec. Recall F1\t   nOk\t  nErr\t nMiss\tFilename\n"
		for taskName, nOk, nErr, nMiss, ltisRefsRunbErrbMiss in lltisRefsRunbErrbMiss:		
			if taskName == None: taskName = ""
			sRpt = """
			<hr/>
			<h2>%s</h2>
<table>
	<tr align="left">
		<th></th>
		<th>Page</th>
		<th>Reference</th>
		<th>Run</th>
		<th></th>
	</tr>
"""		% taskName
			fHtml.write(sRpt)
			ipnum_prev = None

			for (ipnum, sRef, sRun, bErr, bMiss) in ltisRefsRunbErrbMiss:
				if bErr and bMiss:
					sRptType = "Error+Miss"
				else:
					if bErr:
						sRptType = "Error"
					elif bMiss:
						sRptType = "Miss"
					else:
						sRptType = "OK"
					
				if ipnum:	
					sPfFile = sCollec + "/" + sFile + "/" + "pf%06d"%ipnum
				else:
					sPfFile = sCollec + "/" + sFile #may not work!!
					ipnum = ""
					
				lViews = [
						    ('pdf', "/v/" + sCollec + "/" + sFile + "/" + str(ipnum))  #this page deals with generating the real filename
						  , ('dat',  "/dat" + "/" + sPfFile)
						  , ('ocr',  "/ocr" + "/" + sPfFile)
						  , ('xml',  "/xml" + "/" + sPfFile)
						  ]
				lsViews = [ '<a target="dla_%s" href="%s">%s</a>'%(name, url, name) for (name, url) in lViews ]
				
				if ipnum > ipnum_prev: #a new page
					fHtml.write('<tr class="%s"><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>\n' % (sRptType, sRptType
								, ipnum
								, sRef
								, sRun
								, " - ".join(lsViews)
								))
				else: #some more results for the same pafe
					fHtml.write('<tr class="%s"><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>\n' % (sRptType, sRptType
								, ""
								, sRef
								, sRun
								, ""
								))
				ipnum_prev = ipnum
				
			fHtml.write('</table>')
			fHtml.write('<p/>')
			fHtml.write( self.genHtmlTableReport(sCollec, None, [(filename, nOk, nErr, nMiss, None)] ) )

		fHtml.write('<hr>')
		
		fHtml.close()
		
		return

	
	##Compute the precision, recall, and F1 given the number of ok, errors, misses
	#@param nOk: number of good decisions
	#@param nErr: number of wrong positive decisions
	#@param nMiss: number of wrong negative decisions
	#@return (Precision, Recall, F1) as a tuple of float
	#***undefined values are return as None - see formatPRF method! ***
	def computePRF(nOk, nErr, nMiss):
		try:
			fP = float(nOk) / (nOk+nErr)
		except ZeroDivisionError:
			fP = None
		try:
			fR = float(nOk) / (nOk+nMiss)
		except ZeroDivisionError:
			fR = None
			
		try:
			fF = 2 * fP * fR / (fP + fR)
		except ZeroDivisionError:
			fF = 0.0
		except TypeError:
			fF = None
		
		return fP, fR, fF
	computePRF = staticmethod(computePRF)
	
	def formatFloat(f, d=1, sNotApplicable="N/A"):
		"""
		format a float with n decimals (1 decimal per default)
		return "N/A" if the float equals None
		"""
		if f == None:
			return sNotApplicable
		else:
			sFmt = "%%.%df"%d
			return sFmt%f
	formatFloat = staticmethod(formatFloat)
		
	def formatFloatPercent(f, d=1, sNotApplicable="N/A"):
		"""
		format a float with n decimals (1 decimal per default)
		return "N/A" if the float equals None
		"""
		if f == None:
			return sNotApplicable
		else:
			sFmt = "%%.%df"%d
			return sFmt%(100.0*f)
	formatFloatPercent = staticmethod(formatFloatPercent)

	##format nicely a tuple (precision, recall, F1)
	#@param fP: precision
	#@param fR: recall
	#@param fF: F1
	#@param bPlusMoins OPTIONAL
	#@return (Precision, Recall, F1) as a tuple of float
	def formatPRF(fP, fR, fF, bPlusMoins=False):
		if bPlusMoins:
			#sPlusMoins = u"\u00B1"
#			sPlusMoins = "+-"
#			sPlusMoins = '\xb1'
			sPlusMoins = '~'
		else:
			sPlusMoins = " "
		n = 7
		spc=" "*7 #eclipse IDE is used to replace seuqneces of space by tabs...
		s = ""
		s += (spc+"%s%s"%(sPlusMoins, Component.formatFloatPercent(fP, 1)))[-n:]
		s += (spc+"%s%s"%(sPlusMoins, Component.formatFloatPercent(fR, 1)))[-n:]
		s += (spc+"%s%s"%(sPlusMoins, Component.formatFloatPercent(fF, 1)))[-n:]
		return s
	formatPRF = staticmethod(formatPRF)
	
	# Compute average and standard deviation of a list of floating point values
	# Return ( <count> , <average>, <standard-deviation> )
	def av_stddev_f( lFloat ):
		if not lFloat:
			return 0, 0, 0
	
		sx = 0.0
		sx2 = 0.0
	
		#Compute CNT, SX, SX2
		cnt = len( lFloat )
		for v in lFloat:
			if v != None:
				sx = sx + v
				sx2 = sx2 + v*v
	
		#Compute average and standard deviation
		if cnt == 0:
			mx = 0.0
			sdx = 0.0
		else:
			mx = float(sx) / cnt
			sdx = math.sqrt( float(sx2)/cnt - mx * mx );
		return ( cnt, mx, sdx )
	av_stddev_f = staticmethod(av_stddev_f)
	
 
	#------------- META-TAG FOR TRACEABILITY -------------------
	
	## Add a <PROCESS> tag into the <METADATA> tag in the XML document input.
	# <METADATA><br></br>
	# &nbsp;<PDFFILENAME>*****.pdf</PDFFILENAME><br></br>
	# &nbsp;<VERSION>pdftoxml version 1.0</VERSION><br></br>
	# &nbsp;<PROCESS name="tool.py" cmd="....."><br></br>
	# &nbsp;&nbsp;<VERSION value="v1.1"><br></br>
	# &nbsp;&nbsp;&nbsp;<COMMENT>.....</COMMENT><br></br>
	# &nbsp;&nbsp;</VERSION><br></br>
	# &nbsp;&nbsp;<CREATIONDATE>Tue Oct 24 08:30:27 2006</CREATIONDATE><br></br>
	# &nbsp;</PROCESS><br></br>
	# </METADATA></code>
	#@param doc The document object
	def addTagProcessToMetadata(self, doc=None):
		if not doc: doc = self.loadDom()
		ctxt = doc.xpathNewContext()
		ctxt.setContextNode(doc.getRootElement())
		metadata= ctxt.xpathEval('/*/%s[1]'%ds_xml.sMETADATA)
		ctxt.xpathFreeContext()
		if len(metadata)>=1:
			meta = metadata[0].newChild(None, ds_xml.sPROCESS, None)
			#show both the name, sys.argv[0[ and the args, because a component can be activated from anothe rone
			# and it is useful to see this from the PROCESS tags.
			#JL Dec. 07
			meta.setProp(ds_xml.sATTR_NAME, self.name)
			meta.setProp(ds_xml.sATTR_CMD, self.getProgName())
			meta.setProp(ds_xml.sATTR_CMD_ARG, self.buildCmdLine())
			version = meta.newChild(None, ds_xml.sVERSION, None)
			version.setProp(ds_xml.sATTR_VALUE, self.versionComponent)
			comment = version.newChild(None, ds_xml.sCOMMENT, None)
			meta.newChild(None, ds_xml.sCREATIONDATE, time.ctime())
		else:
			self.addTagMetadata(doc)
			self.addTagProcessToMetadata(doc)

	def hasTagProcess(self, doc):
		"""
		does this DOM have the PROCESS tag of this component??
		if yes return the list of nodes PROCESS with the appropriate name
		if no returns []
		"""
		ctxt = doc.xpathNewContext()
		ctxt.setContextNode(doc.getRootElement())
		lNd = ctxt.xpathEval('/*/%s/%s[@%s="%s"]'%(ds_xml.sMETADATA, ds_xml.sPROCESS, ds_xml.sATTR_NAME, self.name))
		ctxt.xpathFreeContext()
		return lNd
	hasTagProcess = classmethod(hasTagProcess)
	
	## Build the command line with each parameter in the list
	# (Note : the boolean parameters are ignored)
	#@return the command line like a string
	def buildCmdLine(self):
		return `self.dParam`
		
	## Add a <METADATA> tag into the <DOCUMENT> tag in the XML document input.
	#@param doc The document object
	def addTagMetadata(self, doc):
		node = doc.getRootElement()
		first = node.children
		if first: first.addPrevSibling(libxml2.newNode(ds_xml.sMETADATA))
		else: node.newChild(None, ds_xml.sMETADATA, None)

	## Get the program python name which has been executed
	#@return the program python name
	def getProgName(self):
		return sys.argv[0]   
	
	#---- UTILITIES ----
	## Get the arguments
	#@return the arguments
	def getParams(self):
		return self.dParam
	
	## Get the usage of the component
	#@return the usage in a string
	def getUsageComponent(self):
		return self.usageComponent
			
	## Modify the version value
	#@param version The new version value
	def setVersion(self, version):
		self.parser.version=version
		
	## Get the version value
	#@return the version		
	def getVersion(self):
		return self.versionComponent
		
	## Begin the chrono
	def beginChrono(self):
		self.main_chrono = chrono.Chrono().on()
		
	## Get the current value of the chrono and print the result if msg is not None
	#@param msg the message to be display after the duration (in second)
	#@return the duration in second as a float
	def getChrono(self, msg=None):
		fDuration = self.main_chrono.get()
		if msg != None: traceln("[%.1fs] %s"% (fDuration, msg))   
		return fDuration 

	## End the chrono and print the result is msg is not None
	#@param msg the message to be display after the duration (in second)
	#@return the duration in second as a float
	def endChrono(self, msg=None):
		fDuration = self.main_chrono.off()
		if msg != None: traceln("[%.1fs] %s"% (fDuration, msg))   
		return fDuration   
	
	## Turn On/OFF the verbose mode
	#@return a boolean (self.bVerbose)
	def setVerbose(self, b):
		self.bVerbose = b
		return b

	## Is this component in verbose mode?
	#@return a boolean (self.bVerbose)
	def isVerbose(self):
		return self.bVerbose

	## Turn On/OFF the debug mode
	#@return a boolean (self.bDebug)
	def setDebug(self, b):
		self.bDebug = b
		return b

	## Is this component in debug mode?
	#@return a boolean (self.bDebug)
	def isDebug(self):
		return self.bDebug

	## Turn On/OFF the zlib mode
	#@return a boolean (self.bDebug)
	def setZLib(self, b):
		self.bZLib = b
		return b

	## Is this component in debug mode?
	#@return a boolean (self.bDebug)
	def isZLib(self):
		return self.bZLib

##
# the Component.py exception
class ComponentException(Exception):
	pass


		
if __name__=="__main__":
	## SELF-TEST
	cp = Component("je m'appelle JoJo", "test-component", "1", "self-test")
	cp.createCommandLineParser()
	dParams = cp.parseCommandLine()
	#cp.setArgs(dParams)
#	print dParams

	#test of the introspection use for the testRun method
	class A:
		def __init__(self):
			pass
	
		def testDir(self, x, y=None):
			
			argspec = inspect.getargspec(self.testRun)
			nbargs = len(argspec.args)
			assert nbargs in (2,3), "internal error: testRun method must have either 2 or  arguments"
			if nbargs == 2:
				ret = self.testRun(x)
			else:
				ret = self.testRun(x, y)
			return ret
			
	class B(A):
		def __init__(self):
			pass
		
		def testRun(self, x):
			print "B.testRun(%s)"%x
			return (x,)
			
	class C(A):
		def __init__(self):
			pass
		def testRun(self, x,y):
			print "B.testRun(%s,%s)"%(x,y)
			return (x,y)
			
	class D(A):
		def __init__(self):
			pass
		def testRun(self, x,y=None):
			print "D.testRun(%s,%s)"%(x,y)
			if y == None:
				return (x,)
			else:
				return (x,y)
	
	b=B()
	assert b.testDir(1)   == (1,)
	
	c=C()
	assert c.testDir(2,3) == (2,3)    
	
	d=D()
	assert d.testDir(2,3) == (2,3)
	assert d.testDir(9)   == (9,)
