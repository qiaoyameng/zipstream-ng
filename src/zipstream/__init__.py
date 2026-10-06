#!/usr/bin/env python

# Make all functionality available in the main `zipstream` namespace
from zipstream.ng import __all__, __doc__
from zipstream.ng import *

# Multi-volume (spanned) archive support
from zipstream.spanned import (
    SpannedZipStream, ZipVolume, SpannedZipError, VolumeSizeError,
)

__all__ += [
    "SpannedZipStream", "ZipVolume", "SpannedZipError", "VolumeSizeError",
]
