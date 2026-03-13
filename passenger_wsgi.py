import sys
import os

project_home = '/home/USERNAME/public_html/Countrylink-management-system-main'

if project_home not in sys.path:
    sys.path.insert(0, project_home)

from app import app as application
