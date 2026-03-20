import sys, os

venv_activate = '/home/countrylinks/virtualenv/public_html/Countrylink-management-system-main/3.13/bin/activate_this.py'
if os.path.exists(venv_activate):
    with open(venv_activate) as f:
        exec(f.read(), {'__file__': venv_activate})

project_home = os.environ.get('PASSENGER_APP_ROOT', '/home/countrylinks/public_html/Countrylink-management-system-main')
if not os.path.isdir(project_home):
    project_home = os.path.dirname(os.path.abspath(__file__))
if project_home not in sys.path:
    sys.path.insert(0, project_home)

try:
    from app import app as application
except Exception:
    import traceback
    _tb = traceback.format_exc()
    def application(environ, start_response):
        body = ('<h2>App failed to start</h2><pre>' + _tb + '</pre>').encode()
        start_response('500 Internal Server Error', [
            ('Content-Type', 'text/html'),
            ('Content-Length', str(len(body)))
        ])
        return [body]
