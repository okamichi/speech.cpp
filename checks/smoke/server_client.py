"""A client of `speech serve` for the smoke scripts: starts it on a free port, waits until it answers, and stops it when
the script ends, however it ends, so that a failed check leaves no server behind; and sends requests, and speaks over a
WebSocket (RFC 6455), with the standard library alone.
"""

import atexit
import base64
import hashlib
import http.client
import json
import os
import re
import socket
import struct
import subprocess
import threading
import time
import uuid


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, speech, args, env=None, host="127.0.0.1"):
        self.port = free_port()
        self.process = subprocess.Popen([speech, "serve", *args, "--port", str(self.port), "--host", host], stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, env=env)
        atexit.register(self.stop)
        # stderr is read as the server writes it, since a log that fills the pipe would stop the server. The reader is a
        # daemon: Python joins the other threads before it runs atexit, and this one ends only once the server has.
        self.lines = []
        self.reader = threading.Thread(target=lambda: self.lines.extend(line.decode(errors="replace") for line in self.process.stderr),
                                       daemon=True)
        self.reader.start()
        t0 = time.perf_counter()
        while True:
            if self.process.poll() is not None:
                self.reader.join()
                raise SystemExit(f"the server exited with {self.process.returncode}: {''.join(self.lines)[-600:]}")
            try:
                if self.call("GET", "/health")[0] == 200:
                    break
            except OSError:
                time.sleep(0.2)
        self.started = time.perf_counter() - t0

    @property
    def page(self):
        """The page's address as the server printed it before it listened, or None where it has no page."""
        for _ in range(50):
            for line in self.lines:
                # The page's address ends in "/", and the API's, where the server has no page, in "/v1".
                found = re.search(r"Ready at (http://\S+)$", line.strip())
                if found:
                    return found.group(1) if found.group(1).endswith("/") else None
            time.sleep(0.1)
        return None

    def logged(self):
        """The lines of the log so far, each without the time of day it begins with."""
        return [re.sub(r"^\d\d:\d\d:\d\d  ", "", line) for line in list(self.lines)]

    def wait_for_line(self, wanted, what):
        """Waits for a line of the log, without its time of day, that `wanted` accepts; fails naming `what` after 5 s."""
        for _ in range(50):
            if any(wanted(line) for line in self.logged()):
                return
            time.sleep(0.1)
        raise SystemExit(f"the log has no line of {what}: " + "".join(list(self.lines)[-4:]))

    def call(self, method, path, body=None, headers=None, stream=False):
        """The status, the headers by lower-case name and the body, or the open response of a stream."""
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=600)
        c.request(method, path, body=body, headers=headers or {})
        r = c.getresponse()
        if stream:
            return r
        data = r.read()
        c.close()
        return r.status, {k.lower(): v for k, v in r.getheaders()}, data

    def websocket(self, path, headers=None):
        """A WebSocket opened at `path`, or the status, the headers by lower-case name and the body of a refused upgrade."""
        return WebSocket.open("127.0.0.1", self.port, path, headers or {})

    def post_json(self, path, member, headers=None):
        return self.call("POST", path, json.dumps(member, ensure_ascii=False).encode(), {"Content-Type": "application/json", **(headers or {})})

    def transcribe(self, fields, files, headers=None):
        """POST /v1/audio/transcriptions with the form's fields, (name, value), files, (name, filename, content), and headers."""
        boundary = uuid.uuid4().hex
        body = b""
        for name, value in fields:
            body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
        for name, filename, content in files:
            body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n"
                     f"Content-Type: audio/wav\r\n\r\n").encode() + content + b"\r\n"
        body += f"--{boundary}--\r\n".encode()
        return self.call("POST", "/v1/audio/transcriptions", body, {"Content-Type": f"multipart/form-data; boundary={boundary}", **(headers or {})})

    def stop(self):
        """Stops the server and returns what it wrote on stdout."""
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.reader.join()
        return self.process.stdout.read()


class WebSocket:
    """A client's end of a WebSocket: text messages out, masked as a client masks them, and messages in, answering pings."""

    GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

    def __init__(self, sock):
        self.sock = sock
        self.buffer = b""

    @classmethod
    def open(cls, host, port, path, headers):
        sock = socket.create_connection((host, port), timeout=600)
        key = base64.b64encode(os.urandom(16)).decode()
        lines = {"Host": f"{host}:{port}", "Upgrade": "websocket", "Connection": "Upgrade", "Sec-WebSocket-Key": key, "Sec-WebSocket-Version": "13",
                 **headers}
        sock.sendall((f"GET {path} HTTP/1.1\r\n" + "".join(f"{k}: {v}\r\n" for k, v in lines.items()) + "\r\n").encode())
        ws = cls(sock)
        head = ws._until(b"\r\n\r\n").decode()
        status = int(head.split(" ", 2)[1])
        answer = {k.strip().lower(): v.strip() for k, v in (line.split(":", 1) for line in head.split("\r\n")[1:] if ":" in line)}
        if status != 101:
            body = ws._exactly(int(answer.get("content-length", "0")))
            sock.close()
            return status, answer, body
        accept = base64.b64encode(hashlib.sha1((key + cls.GUID).encode()).digest()).decode()
        assert answer.get("sec-websocket-accept") == accept, answer
        return ws

    def _fill(self):
        chunk = self.sock.recv(65536)
        if not chunk:
            raise ConnectionError("the server closed the connection")
        self.buffer += chunk

    def _until(self, mark):
        while mark not in self.buffer:
            self._fill()
        head, self.buffer = self.buffer.split(mark, 1)
        return head

    def _exactly(self, n):
        while len(self.buffer) < n:
            self._fill()
        out, self.buffer = self.buffer[:n], self.buffer[n:]
        return out

    @staticmethod
    def _frame_bytes(opcode, payload):
        mask = os.urandom(4)
        n = len(payload)
        head = bytes([0x80 | opcode]) + (bytes([0x80 | n]) if n < 126 else bytes([0x80 | 126]) + struct.pack(">H", n) if n < 65536
                                         else bytes([0x80 | 127]) + struct.pack(">Q", n))
        return head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload))

    def _frame(self, opcode, payload):
        self.sock.sendall(self._frame_bytes(opcode, payload))

    def send_split(self, message, pause):
        """Sends a JSON object as one text frame written in two pieces `pause` seconds apart, as a slow network delivers it."""
        frame = self._frame_bytes(0x1, json.dumps(message, ensure_ascii=False).encode())
        self.sock.sendall(frame[:len(frame) // 2])
        time.sleep(pause)
        self.sock.sendall(frame[len(frame) // 2:])

    def send(self, message):
        """Sends a text message, or a JSON object as one."""
        self._frame(0x1, (message if isinstance(message, str) else json.dumps(message, ensure_ascii=False)).encode())

    def send_binary(self, data):
        self._frame(0x2, data)

    def receive(self):
        """The next message's text, or None once the server has closed the WebSocket."""
        parts = b""
        while True:
            first, second = self._exactly(2)
            n = second & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._exactly(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._exactly(8))[0]
            assert not second & 0x80, "the server masked a frame"
            payload, opcode = self._exactly(n), first & 0x0F
            if opcode == 0x9:
                self._frame(0xA, payload)
            elif opcode == 0x8:
                return None
            elif opcode in (0x0, 0x1, 0x2):
                parts += payload
                if first & 0x80:
                    return parts.decode()

    def event(self):
        """The next message as a JSON object."""
        text = self.receive()
        assert text is not None, "the server closed the WebSocket"
        return json.loads(text)

    def close(self):
        try:
            self._frame(0x8, struct.pack(">H", 1000))
            while self.receive() is not None:
                pass
        except (ConnectionError, OSError):
            pass
        self.sock.close()


def expect_error(got, status, code, param, what):
    """Checks an error in OpenAI's shape, with its status, type, code and param, and prints it."""
    s, _, body = got
    error = json.loads(body).get("error", {}) if body else {}
    kind = "server_error" if status >= 500 else "invalid_request_error"
    if s != status or error.get("code") != code or error.get("param") != param or error.get("type") != kind or not error.get("message"):
        raise SystemExit(f"{what}: expected {status} {code} ({param}), got {s} {body[:300]!r}")
    print(f"{what}: {status} {kind} {code} ({param}) as expected: {error['message'][:90]}")
