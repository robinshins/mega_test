# pylint: skip-file
import os
ENVIRONMENT = os.environ.get('RAG_ENV', 'production').lower()

from .production import *
if ENVIRONMENT == 'development':
    from .development import *
if ENVIRONMENT == 'test':
    from .testing import *
