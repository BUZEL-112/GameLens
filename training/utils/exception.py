"""
Custom exception wrapper.

Preserves the original traceback while adding module context.
"""

import sys


class CustomException(Exception):
    def __init__(self, error: Exception):
        _, _, tb = sys.exc_info()
        if tb is not None:
            frame = tb.tb_frame
            lineno = tb.tb_lineno
            filename = frame.f_code.co_filename
            message = f"[{filename}:{lineno}] {type(error).__name__}: {error}"
        else:
            message = f"{type(error).__name__}: {error}"
        super().__init__(message)
        self.__cause__ = error
