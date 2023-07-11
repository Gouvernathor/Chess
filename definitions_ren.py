import os
import renpy # type: ignore
from renpy import config # type: ignore
"""renpy
init python hide:
"""
# mimics common/00image.rpy but for our own images, and not replacing existing ones
prefix = ""

for fn in renpy.list_files():
    if fn.startswith(prefix):
        base, ext = os.path.splitext(os.path.basename(fn))

        if ext.casefold() in config.image_extensions:
            base = base.lower().partition("@")[0]

            if not renpy.has_image(base, exact=True):
                renpy.image(base, fn)
