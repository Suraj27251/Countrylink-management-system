import sys
import os

# cPanel sometimes keeps a locally edited passenger_wsgi.py path, which can block
# "Update from Remote" if upstream changes this file. Keep the same deploy path
# as the cPanel repository location by default, but still allow override.
project_home = os.environ.get(
    'PASSENGER_APP_ROOT',
    '/home/countrylinks/public_html/Countrylink-management-system-main'
)
if not os.path.isdir(project_home):
    project_home = os.path.dirname(os.path.abspath(__file__))

if project_home not in sys.path:
    sys.path.insert(0, project_home)

from app import app as application
