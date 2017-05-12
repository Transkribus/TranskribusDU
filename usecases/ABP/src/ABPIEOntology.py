# -*- coding: utf-8 -*-
"""

    ABO Death records IEOntology
    Hervé Déjean
    cpy Xerox 2017
    
    death record

    READ project 

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
import sys, os.path
sys.path.append (os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))) + os.sep+'src')
from ObjectModel.recordClass import recordClass,fieldClass, REtaggerClass


class deathRecord(recordClass):
    sName = 'deathrecord' 
    def __init__(self):
        recordClass.__init__(self,deathRecord.sName)
        
    
    
def firstNameField(fieldClass):
    sName = 'firstname'
    def __init__(self):
        fieldClass.__init__(self, fieldClass.sName)


def fnameTagger(REtaggerClass):
    sName= 'firstNameTagger'
    def __init__(self):
        taggerClass.__init__(self, fnameTagger.sName)
        self._typeOfTagger = taggerClass.FSTTYPE
        self.__path="./ressources/firstnames.200.pkl"
        self._mapping={'firstname':firstname}
        
        

def test_ieo():   
    dr =deathRecord()
    fnField= firstNameField()
    myFNameTagger =fnameTagger()
    myFNameTagger.add
    dr.addField(fnField)
    print dr        
    
