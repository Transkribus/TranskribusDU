# -*- coding: utf-8 -*-
"""
    conversion pixel<>point
"""

import types
def convertDot2Pixel(dpi,value):
    assert type(value) in [types.IntType,types.FloatType], type(value)
    assert dpi != 0.0, 'DPI value can not be Zero'
    return int(round(1.0*value*(dpi/72.0),0))


def convertPixel2Dot(dpi,value):
    assert type(value) in [types.IntType,types.FloatType], type(value)
    assert dpi != 0.0
    return round(1.0*value*(72.0/dpi),-2)