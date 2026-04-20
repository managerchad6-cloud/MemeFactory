#!/usr/bin/env python3
"""
Serves the MemeFactory UI on port 8001.
"""
import http.server
import socketserver
import os

PORT = 8001
DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIR, **kwargs)

    def log_message(self, fmt, *args):
        print(f"[UI] {self.address_string()} - {fmt % args}")


with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
    print(f"MemeFactory UI → http://0.0.0.0:{PORT}")
    httpd.serve_forever()
