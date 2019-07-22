"""
Parameter class
A parameter contains a name, a short option, a long option and an argument

Sophie Andrieu - October 2006

Copyright Xerox XRCE 2006

"""

import types
from common.trace import  traceln

## The <code>Parameter</code> class describes a parameter defined by a name, a short option, a long option and an argument.
# It can also contains a default value, a type and a description.
#@version 1.1
#@date October 2006
#@author Sophie Andrieu - Copyright Xerox XRCE 2006

class Parameter:
    
    type = None
    defaultValue = None
    description = None
    
    ## Build a new <code>Parameter</code> object
    #@param name The name of the parameter
    #@param optionShort The short option value
    #@param optionLong The long option value
    #@param arg The argument value
    def __init__(self, name = None, optionShort = None, optionLong = None, arg = None):
        self.name = name
        self.optionLong = optionLong
        self.optionShort = optionShort
        self.arg = arg
        
    ## Modify the long option parameter
    #@param optLong The new long option
    def setOptionLong(self, optLong):
        self.optionLong = optLong
        
    ## Modify the short option parameter
    #@param optShort The new short option
    def setOptionShort(self, optShort):
        self.optionShort = optShort
        
    ## Modify the name parameter
    #@param name The new name parameter
    def setName(self, name):
        self.name = name
        
    ## Modify the argument value parameter
    #@param arg The new argument value parameter  
    def setArg(self, arg):
        self.arg = arg

    ## Modify the type parameter
    #@param type The new type parameter  
    def setType(self, type):
        self.type = type

    ## Modify the default value parameter
    #@param default The new default value parameter  
    def setDefaultValue(self, default):
        self.defaultValue = default

    ## Modify the description parameter
    #@param desc The new description parameter  
    def setDescription(self, desc):
        self.description = desc
        
    ## Get the long option parameter
    #@return The long option
    def getOptionLong(self):
        return self.optionLong

    ## Get the short option parameter
    #@return The short option 
    def getOptionShort(self):
        return self.optionShort

    ## Get the argument value parameter
    #@return The argument value
    def getArg(self):
        return self.arg

    ## Get the name parameter
    #@return The name
    def getName(self):
        return self.name

    ## Get the type parameter
    #@return The type
    def getType(self):
        return self.type

    ## Get the default value parameter
    #@return The default value
    def getDefaultValue(self):
        return self.defaultValue

    ## Get the description parameter
    #@return The desciption
    def getDescription(self):
        return self.description
    
    ## Get the boolean which represent the string value
    #@return <code>True</code> if the string value is 'True', <code>False</code> if the string value is 'False' 
    def getBoolValue(self, argString):
        if argString == "True":
            return True
        elif argString == "False":
            return False
        
    ## Get a string which represents the command line for this parameter
    # (Note : the boolean parameters are ignored)
    #@return the command line like a string
    def getStringCmdLine(self):
        long = False  
        result = ""
        option = ""
        if self.optionShort:
            option = self.optionShort
        elif self.optionLong:
            option = self.optionLong
            long = True
        if self.arg:
            if self.arg!="":
                if type(self.arg) != types.BooleanType:
                    if long:
                        result = result + option + "=" + str(self.arg) + " "
                    else: 
                        result = result + option + " " + str(self.arg) + " "
        return result
                
    ## Get a string value which contains all informations about the current parameter
    #@return a string value
    def toString(self):
        traceln("{name=", self.name, ", short=", self.optionShort, ", long=", self.optionLong, ", arg=", self.arg, "}")
        traceln("{type=", self.type, ", desc=", self.description, ", default=", self.defaultValue, "}")
    