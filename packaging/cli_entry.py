# SPDX-License-Identifier: Apache-2.0
"""PyInstaller entry for FBTPods-console.exe (terminal: diag, pair, demo, run)."""
import sys

from airpodsvr.__main__ import main

if __name__ == "__main__":
    main(sys.argv[1:])
