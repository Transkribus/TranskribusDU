# -*- coding: utf-8 -*-

"""
    Compare two detailled Report ()
    
    Copyright Naber Labs Europe(C) 2018 
    @author H. Déjean

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
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""




import sys,os
from optparse import OptionParser
import pickle, gzip
sys.path.append( os.path.dirname(os.path.dirname( os.path.abspath(sys.argv[0]) )) )

from common.TestReport import TestReport

usage=""
version="1.0"

parser = OptionParser(usage=usage, version=version)
parser.add_option("--report1"  , dest='report1', action="store", type="string", help="Report 2 filename")
parser.add_option("--report2"  , dest='report2', action="store", type="string", help="Report 2 filename")
parser.add_option("--trace"  , dest='trace', action="store_true", default=False, help="print sample")

(options, args) = parser.parse_args()

with gzip.open(options.report1, "rb") as zfd:
    r1 =pickle.load(zfd)
       
with gzip.open(options.report2, "rb") as zfd:
    r2 =pickle.load(zfd)       

TestReport.compareReport(r1,r2,options.trace)

