import types
import shapely.geometry as geom


from common.trace import traceln
from util.Polygon import Polygon

from .NodeType import NodeType
from .Block import Block
from .Page import Page
#from PIL import Image


def defaultBBoxDeltaFun(w):
    """
    When we reduce the width or height of a bounding box, we use this function to compute the deltaX or deltaY
    , which is applied on x1 and x2 or y1 and y2

    For instance, for horizontal axis
        x1 = x1 + deltaFun(abs(x1-x2))
        x2 = x2 + deltaFun(abs(x1-x2))
    """
    # "historically, we were doing:
    dx = max(w * 0.066, min(20, w / 3))
    # for ABP table RV is doing: dx = max(w * 0.066, min(5, w/3)) , so this function can be defined by the caller.
    return dx



class NodeType_jsonOCR(NodeType):
    # where the labels can be found in the data
    sCustAttr_STRUCTURE = "structure"
    sCustAttr2_TYPE = "type"


    def __init__(self, sNodeTypeName, lsLabel, lsIgnoredLabel=None, bOther=True, BBoxDeltaFun=defaultBBoxDeltaFun
                 , bPreserveWidth=False
                 ):
        NodeType.__init__(self, sNodeTypeName, lsLabel, lsIgnoredLabel, bOther)

        self.BBoxDeltaFun = BBoxDeltaFun
        if self.BBoxDeltaFun is not None and type(self.BBoxDeltaFun) != types.FunctionType:
            raise Exception("Error: BBoxDeltaFun must be None or a function (or a lambda)")
        self.bPreserveWidth = bPreserveWidth

    def setXpathExpr(self, t_sxpNode_sxpTextual):
        """
        generalisation of XPATH to JSON format?
        """
        raise Exception("Not yet implemented")

    def getXpathExpr(self):
        """
        generalisation of XPATH to JSON format?
        """
        raise Exception("Not yet implemented")

    def parseDocNodeLabel(self, graph_node, defaultCls=None):
        """
        Parse and set the graph node label and return its class index
        raise a ValueError if the label is missing while bOther was not True, or if the label is neither a valid one nor an ignored one
        """
        raise Exception("Not yet implemented")

    def setDocNodeLabel(self, graph_node, sLabel):
        """
        Set the DOM node label in the format-dependent way
        """
#         print("setDocNodeLabel "
#               , "%s '%s'" %(graph_node.getShape(), graph_node.text)
#               , " ", sLabel)
        pass
        # raise Exception("Not yet implemented")

    def setLabelAttribute(self, sAttrName="type"):
        """
        set the name of the Xml attribute that contains the label
        """
        self.sLabelAttr = sAttrName

    def getLabelAttribute(self):
        return self.sLabelAttr

    # ---------------------------------------------------------------------------------------------------------

#     def getPageWidthandHeight(self, filename):
# 
#         img_dir = '/nfs/project/nmt/menus/data/GT500/'
#         file = img_dir + filename.split('/')[-1][:-4] + 'jpg'
#         im = Image.open(file)
#         return im.size
    # ---------------------------------------------------------------------------------------------------------

    def getPointList(self, ndBlock):
        coords = ndBlock['boundingBox']
        x = coords[0]
        y = coords[1]
        w = coords[2]
        h = coords[3]
        return [(x,y), (x+w, y),(x+w, y+h), (x, y+h)]
    # ----------------------------------------------------------------------------------------------------------

    def _iter_GraphNode(self, doc, sFilename, page=None):
        """
        Get the json dict

        iterator on the DOM, that returns nodes  (of class Block)
        """
        # --- XPATH contexts

        lNdBlock = doc['GlynchResults']['Areas']
        #page_w, page_h = self.getPageWidthandHeight(sFilename)
        page = Page(1, 1, 1, 1)
        for ndBlock in lNdBlock:
            try:
                sText = ndBlock['label']
            except KeyError:
                sText = ""
                traceln("WARNING: no 'label' in : %s" % ndBlock)
            if sText == None:
                sText = ""
                traceln("Warning: no text in node")
                # raise ValueError, "No text in node: %s"%ndBlock

            # now we need to infer the bounding box of that object
            lXY = self.getPointList(ndBlock)  # the polygon
            if lXY == []:
                continue

            plg = Polygon(lXY)
            try:
                x1, y1, x2, y2 = plg.fitRectangle(bPreserveWidth=self.bPreserveWidth)
            except ZeroDivisionError:
                x1, y1, x2, y2 = plg.getBoundingBox()
            except ValueError:
                x1, y1, x2, y2 = plg.getBoundingBox()

            # we reduce a bit this rectangle, to ovoid overlap
            if not (self.BBoxDeltaFun is None):
                if type(self.BBoxDeltaFun) is tuple and len(self.BBoxDeltaFun) == 2:
                    xFun, yFun = self.BBoxDeltaFun
                    if xFun is not None:
                        dx = xFun(x2-x1)
                        x1, x2 = int(round(x1+dx)), int(round(x2-dx))
                    if yFun is not None:
                        dy = yFun(y2-y1)
                        y1, y2 = int(round(y1+dy)), int(round(y2-dy))
                else:
                    # historical code
                    w, h = x2 - x1, y2 - y1
                    dx = self.BBoxDeltaFun(w)
                    dy = self.BBoxDeltaFun(h)
                    x1, y1, x2, y2 = [int(round(v)) for v in [x1 + dx, y1 + dy, x2 - dx, y2 - dy]]

            # TODO
            orientation = 0  # no meaning for PageXml
            classIndex = 0  # is computed later on
            # and create a Block
            blk = Block(page, (x1, y1, x2 - x1, y2 - y1), sText, orientation, classIndex, self, ndBlock, domid=None)
            blk.setShape(geom.Polygon(lXY))
            yield blk

        return
