import os

DEBUG = True

BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../')
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_PATH, 'db.sqlite3'),
    }
}
