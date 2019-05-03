import sys, wx, os.path

if wx.VERSION == (2, 6, 1, 0, ''):
	print "*"*40
	print " FATAL ERROR: OBSOLETE WX Version: ", wx.VERSION
	print "*"*40
	sys.exit(1)
	
"""
TODO: deal properly with images (they are in negative.... :-( )

"""

import os.path
import config
import MyFrame
import deco

sEncoding = "utf-8"



class MyApp(wx.App):
	
	def __init__(self, arg, sFile = None):
		self.sFile = sFile
		wx.App.__init__(self, arg)
		
	def OnInit(self):
		wx.InitAllImageHandlers()
		frame = MyFrame.MyFrame(None, -1, "")		
		self.SetTopWindow(frame)
		frame.Show()
		
		#for an easy dev
		#if sFile and os.path.exists(sFile): 
		if self.sFile : frame.loadXML(self.sFile)
#		if os.path.exists("launiversitat20_noDS.xml"): frame.loadXML("brill92.fn.xml")
#		if os.path.exists("launiversitat20.xml"): frame.loadXML("launiversitat20.xml")
#		if os.path.exists("launiversitat1-5.xml"): frame.loadXML("launiversitat1-5.xml")
		return 1

def lookForConfig(sFile, lsPath):
	for sPath in lsPath:
		s = sPath+os.sep+sFile
		print "Looking for ", s,
		if os.path.exists(s):
			print "OK"
			return s
		else:
			print "FAILED"
	print "No config file %s in %s" % (sFile, lsPath)
	sys.exit(1)
	
if __name__ == "__main__":
	deco.setEncoding(sEncoding)
	sFile = None
	print sys.argv
	
	try:
		sPath = sys.argv[1]
		
		try:
			sFile = sys.argv[2]
			if not os.path.exists(sFile) and len(sys.argv)>3:
				#maybe we have a space in the file name... :-( and it is a mess with windows and .bat
				if os.path.exists(sys.argv[2] + " " + sys.argv[3]):
					sFile = sys.argv[2] + " " + sys.argv[3]
		except IndexError:
			pass
	except IndexError:
		sConfigFileName = "wxvisu.ini"
		#try to find da config.ini file
		sPath = lookForConfig(sConfigFileName, [".", os.path.dirname(os.path.abspath(sys.argv[0]))])
	
	MyFrame.setConfigFile(sPath)
	MyFrame.setEncoding(sEncoding)
	app = MyApp(0, sFile)
	app.MainLoop()
