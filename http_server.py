#!/usr/bin/env python3
from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import json

import numpy as np
from joblib import load
import pandas as pd

class S(BaseHTTPRequestHandler):
    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = self.rfile.read(content_length) # <--- Gets the data itself
        if(str(self.path) != "/score"):
            return
        logging.info("POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n",
                str(self.path), str(self.headers), post_data.decode('utf-8'))
        input_data = json.loads(post_data.decode('utf-8'))
        data = pd.json_normalize(input_data)
        output = self.score(data)
        self._set_response()
        self.wfile.write(f'{output}\n'.encode('utf-8'))
        
    def score(self, data):
        clf = load('log_model.joblib') 
        data['x3'] = data['x3'].str.slice(0, 3)
        tprobs = clf.predict_proba(data)[:, 1]
        return tprobs


def run(server_class=HTTPServer, handler_class=S, port=8080):
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info('Starting httpd...\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info('Stopping httpd...\n')

if __name__ == '__main__':
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()
