"""``splendor_ai`` — importable straight from the repo root.

This outer directory is the project folder (README, requirements, validation
harness, tests) and ``splendor_ai/splendor_ai`` is the library proper.  The
package path lists both so that ``splendor_ai.rules`` resolves to the library
while ``splendor_ai.validation`` and ``splendor_ai.tests`` resolve to the
sibling directories.
"""

import os as _os

_here = _os.path.dirname(__file__)
__path__ = [_os.path.join(_here, "splendor_ai"), _here]

__version__ = "0.1.0"
