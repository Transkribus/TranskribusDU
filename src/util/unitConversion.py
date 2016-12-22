# -*- coding: utf-8 -*-
"""
    conversion pixel<>point
"""

def convertDot2Pixel(dpi,value):
    assert type(value) in ['int','float']
    return int(round(1.0*value*(dpi/72.0),0))


def convertPixel2Dot(dpi,value):
    assert type(value) in ['int','float']
    assert dpi != 0.0
    return round(1.0*value*(72.0/dpi),-2)