"""Lets the command run as `python -m knowcode ...` as well as `knowcode ...`."""

import sys

from knowcode.cli import main

sys.exit(main())
