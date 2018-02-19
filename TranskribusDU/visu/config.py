"""
  
$author: Jerome Fuselier, JL Meunier
$since: July 2006


a Config file looks like:

[General]
working_dir=<where to look for XML file by default>
page_tag=PAGE

[MyDeco]
#this is a new type of decoration I like. Its surname will be MyDeco
#it uses the DecoRectangle type of decoration
type=DecoRectangle

#my deco reflects any TOKEN
xpath=//PAGE//TOKEN

#for this type of decoration, I must tell where to find the X,Y,W,H with some xpath expresssion, relative to the selected node
x=@x
y=@y
w=@width
h=@height



"""

import MyConfigParser, re

import deco

class Config:
    """Keep the main configuration options of the application (global constants)
    """
    
    sSectionGeneral = "General"
   
    def __init__(self, config_file):
        """Default constructor.
        @param config_file: path of the configuration file.
        @type 
        config_file: str
        """ 
        self.sFileName = config_file
        
        print "--- Loading configuration"
        #Some section are pre-defined, or reserved, others are specifi to each usage
        lReservedSections = [ self.sSectionGeneral ] 
        
        self.cfg = MyConfigParser.ConfigParser()
        self.cfg.read(config_file)
        self.cfg.jl_hack_cfg = self #ugly hack due to a slightly bad design (should not pass cfg to decos etc but rather this Config object)
        
        #[General] section
        self.working_dir = self.cfg.get(self.sSectionGeneral, "working_dir")
#        self.encoding    = self.cfg.get(sSectionGeneral, "encoding")
        self.page_tag             = self.getPageTag()
        self.page_tag_attr_number = self.getPageNumberAttr()
        self.page_tag_attr_width  = self.cfg.get(self.sSectionGeneral, "page_tag_attr_width")
        self.page_tag_attr_height = self.cfg.get(self.sSectionGeneral, "page_tag_attr_height")
        self.page_background_color = self.cfg.get(self.sSectionGeneral, "page_background_color")
        self.page_border_color     = self.cfg.get(self.sSectionGeneral, "page_border_color")
        
        #find any namespace declaration
        self.ltNSNameURI = []
        cre = re.compile("xmlns:(.+)")
        for sNSName, sNSURI in self.cfg.items(self.sSectionGeneral):
            mo = cre.match(sNSName)
            if mo:
                sNSName = mo.group(1).strip()
                if sNSURI[0] == '"': sNSURI = sNSURI[1:]
                if sNSURI[-1] == '"': sNSURI = sNSURI[0:-1]
                print "- NameSpace: %s=%s"%(sNSName, sNSURI)
                self.ltNSNameURI.append( (sNSName, sNSURI) )
        
        #Now loop over other sections, which should define a set of decorations
        decos = self.cfg.get(self.sSectionGeneral, "decos")
        lsDeco = decos.split()
        self.lDeco = [] #a list <decoration>
#        for sect in self.cfg.sections():  #random order!!! :-(
        for sect in lsDeco:
            if sect in lReservedSections: continue
            
            sType = self.cfg.get(sect, "type")      #get the name of the associated Python class
            try:
                cls=deco.Deco.getDecoClass(sType)  #get the associated Python class
                obj = cls(self.cfg, sect, None)                    #let the class creates an instance using the details in the config file
                self.lDeco.append(obj)
                #print "-", str(obj)
            except Exception, e:
                print "ERROR: ", e
        print "--- Done ---"
 
    def getPageTag(self):
        return self.cfg.get(self.sSectionGeneral, "page_tag")
    
    def getPageNumberAttr(self):
        try:
            return self.cfg.get(self.sSectionGeneral, "page_tag_attr_number")
        except:
            return "@number"
        
    def getDecoList(self):
        return self.lDeco
    
    def getNamespaceList(self):
        return self.ltNSNameURI
    
    def setXPathContext(self, xpCtxt):
        for deco in self.lDeco:
            deco.setXPathContext(xpCtxt)
    
#        lOption = [ ('working_dir','str', None) ]
#        err_msg = 'Config file not well formated'
#        options = [('bg_image','bool'), 
#                   ('working_dir','str'), ('is_proxy', 'bool'),
#                   ('http_proxy', 'str'), ('http_port', 'str')]
#        
#        config = ConfigParser.ConfigParser()
#        config.read(config_file)        
#        sections = config.sections()
#        
#        gen_opt = 'Options' 
#        if not gen_opt in sections:
#            print "%s : missing '%s' section" % (err_msg, gen_opt)
#        else:
#            for opt in options:
#                if config.has_option(gen_opt, opt[0]):
#                    val = config.get(gen_opt, opt[0])
#                    if opt[1] == 'bool':
#                        val = (val == '1')
#                    elif opt[1] == 'int':
#                        val = int(val)
#                    elif opt[1] == 'float':
#                        val = float(val)
#                    self.__dict__[opt[0]] = val
#                else:
#                    err = "%s : missing %s option" % (err_msg,opt[0])
#                    print err
     
    
if __name__ == "__main__":
    cfg = Config("wxvisu.ini")
    print cfg.getDecoList()
    for d in cfg.getDecoList():
        print str(d)
    
