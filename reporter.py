'''
    Report data to external websocket
'''
import websocket
import json

class Reporter:

    def __init__(self, server_url):
        self.ws = websocket.WebSocket()
        self.ws.connect(server_url)
        
    def report(self, **args):
        payload = json.dumps(args)
        self.ws.send(payload)

    def report_plot(self, plot):
        pass