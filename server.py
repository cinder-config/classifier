import pickle
from http.server import BaseHTTPRequestHandler, HTTPServer
import cgi
import json
import time
import numpy as np

### Webserver Version for predictions

### 1) Define Model & Available Features -> Must match!!
### 2) Send "POST" request containing a json payload with the features to the endpoint `/predict`
### 2.1) Payload is formatted as key -> value feature, eg: 'ch.uzh.ciclassifier.features.configuration.EnvSize': 3

HOSTNAME = "localhost"
PORT = 8081
MODEL_PATH = 'model/classifier.sav'
AVAILABLE_FEATURES = [
    'ch.uzh.ciclassifier.features.configuration.EnvSize',
    'ch.uzh.ciclassifier.features.configuration.LintScore',
    'ch.uzh.ciclassifier.features.configuration.NumberOfComments',
    # 'ch.uzh.ciclassifier.features.configuration.NumberOfConfigurationChars',
    'ch.uzh.ciclassifier.features.configuration.NumberOfConfigurationLines',
    'ch.uzh.ciclassifier.features.configuration.NumberOfJobs',
    'ch.uzh.ciclassifier.features.configuration.NumberOfNotifications',
    'ch.uzh.ciclassifier.features.configuration.NumberOfStages',
    'ch.uzh.ciclassifier.features.configuration.ParseScore',
    'ch.uzh.ciclassifier.features.configuration.UniqueInstructions',
    'ch.uzh.ciclassifier.features.configuration.UseBranches',
    'ch.uzh.ciclassifier.features.configuration.UseCache',
    # 'ch.uzh.ciclassifier.features.github.OwnerType',
    # 'ch.uzh.ciclassifier.features.github.PrimaryLanguage',
    'ch.uzh.ciclassifier.features.repository.CommitsUntilConfigAdded',
    'ch.uzh.ciclassifier.features.repository.ConfigChangeFrequency',
    'ch.uzh.ciclassifier.features.repository.DaysUntilConfigAdded',
    'ch.uzh.ciclassifier.features.repository.NumberOfConfigurationFileChanges',
    'ch.uzh.ciclassifier.features.repository.NumberOfContributors',
    # 'ch.uzh.ciclassifier.features.repository.ProjectName',
    'ch.uzh.ciclassifier.features.travisci.BuildTimeAverage',
    'ch.uzh.ciclassifier.features.travisci.BuildSuccessRatio',
    'ch.uzh.ciclassifier.features.travisci.BuildTimeLatestAverage',
    'ch.uzh.ciclassifier.features.travisci.ManualInteractionRatio',
    'ch.uzh.ciclassifier.features.travisci.PullRequestRatio',
    'ch.uzh.ciclassifier.features.travisci.TimeToFixAverage',
    'ch.uzh.ciclassifier.features.travisci.TimeToFixLatestAverage',
]


def predict(features):
    # load model
    model = pickle.load(open(MODEL_PATH, 'rb'))

    # sanitize (if some non features submitted...)
    sanitized = []
    for feature in AVAILABLE_FEATURES:
        if feature in features:
            sanitized.append(features[feature])

    np_sanitized = [np.array(sanitized)]

    # predict
    predictions = model.predict(np_sanitized)
    probabilities = model.predict_proba(np_sanitized)

    return {'prediction': int(predictions[0]), 'probability': float(probabilities[0][predictions[0]])}


def dispatcher(path, payload):
    if path == '/predict':
        return predict(payload)
    raise Exception('Route does not exist')


class MyServer(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self._set_headers()
        self.end_headers()
        return

    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept")
        self.end_headers()

    def do_POST(self):
        start_time = time.time()
        ctype, pdict = cgi.parse_header(self.headers.get('content-type'))

        if ctype != 'application/json':
            self.send_response(400)
            self.end_headers()
            return
        length = int(self.headers.get('content-length'))
        payload_string = self.rfile.read(length).decode('utf-8')
        payload = json.loads(payload_string) if payload_string else None

        path = self.path
        try:
            message = dispatcher(path, payload)
        except Exception as exept:
            print(str(exept))
            execution_time = time.time() - start_time
            self._set_headers()
            self.wfile.write(bytes(json.dumps({
                'status': 'error',
                'message': str(exept),
                'time': execution_time
            }), encoding='utf8'))
            return

        execution_time = time.time() - start_time
        self._set_headers()
        self.wfile.write(bytes(json.dumps({
            'status': 'success',
            'message': message,
            'time': execution_time
        }), encoding='utf8'))

    def do_GET(self):
        path = self.path
        try:
            message = dispatcher(path, list())
        except Exception as exept:
            print(str(exept))
            self._set_headers()
            self.wfile.write(bytes(json.dumps({
                'status': 'error',
                'message': str(exept)
            }), encoding='utf8'))
            return

        self._set_headers()
        self.wfile.write(bytes(json.dumps({
            'status': 'success',
            'message': message
        }), encoding='utf8'))


webServer = HTTPServer((HOSTNAME, PORT), MyServer)
print("Server started http://%s:%s" % (HOSTNAME, PORT))
webServer.serve_forever()

try:
    webServer.serve_forever()
except KeyboardInterrupt:
    pass

webServer.server_close()
print("Server stopped.")
