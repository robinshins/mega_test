from rag import Application
import settings
from api import models
from api import migrations
from api.urls import urls


# application
app = Application(__name__, settings, urls, models, migrations)
