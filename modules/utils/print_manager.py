import sys
import os


class PrintManager:
    """Class for managing the stdout and stderr"""

    def __init__(self, filter_errors: bool = True) -> None:
        self._filter_errors = filter_errors
        self._stdout = sys.stdout
        self._stderr = sys.stderr

    def disable_print(self) -> None:
        """Block the stdout and stderr"""

        sys.stdout = open(os.devnull, 'w')
        if self._filter_errors:
            sys.stderr = open(os.devnull, 'w')

    def enable_print(self) -> None:
        """Unblock the stdout and stderr"""

        sys.stdout = sys.__stdout__
        if self._filter_errors:
            sys.stderr = sys.__stderr__
