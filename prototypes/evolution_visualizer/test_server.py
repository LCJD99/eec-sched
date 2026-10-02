import importlib.util
import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.request import Request, urlopen


MODULE_PATH = Path(__file__).with_name("server.py")
SPEC = importlib.util.spec_from_file_location("evolution_visualizer_server", MODULE_PATH)
server = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(server)


class TraceServerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.trace_dir = Path(self.tmp.name) / "traces"
        self.trace_dir.mkdir()
        (self.trace_dir / "older.jsonl").write_text(
            '{"sequence":1,"event_type":"session_start","timestamp":"2026-09-20T00:00:00Z","session_id":"old"}\n'
            '{"sequence":2,"event_type":"tool_call","tool":"read_file"}\n', encoding="utf-8"
        )
        (self.trace_dir / "newer.jsonl").write_text(
            '{"sequence":1,"event_type":"session_start","timestamp":"2026-09-21T00:00:00Z","session_id":"new"}\n'
            '{"sequence":2,"event_type":"round_start","round_number":1}\n'
            '{"sequence":3,"event_type":"tool_call","tool":"search"}\n'
            '{"sequence":4,"event_type":"session_end","status":"completed"}\n', encoding="utf-8"
        )
        (self.trace_dir / "notes.txt").write_text("not exposed", encoding="utf-8")
        self.httpd = server.ThreadingHTTPServer(
            ("127.0.0.1", 0),
            lambda *args, **kwargs: server.Handler(*args, trace_dir=self.trace_dir, **kwargs),
        )
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)
        self.tmp.cleanup()

    def get(self, path):
        with urlopen(self.base + path) as response:
            return response.status, response.headers, response.read()

    def assert_not_found(self, path):
        with self.assertRaises(HTTPError) as context:
            urlopen(self.base + path)
        self.assertEqual(context.exception.code, 404)

    def test_trace_list_summary_is_sorted_and_safe(self):
        status, headers, body = self.get("/api/traces")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Cache-Control"], "no-store")
        payload = json.loads(body)
        self.assertEqual([item["filename"] for item in payload["traces"]], ["newer.jsonl", "older.jsonl"])
        self.assertEqual(payload["traces"][0]["event_count"], 4)
        self.assertEqual(payload["traces"][0]["round_count"], 1)
        self.assertEqual(payload["traces"][0]["tool_call_count"], 1)
        self.assertNotIn("notes.txt", [item["filename"] for item in payload["traces"]])

    def test_raw_trace_has_ndjson_content_type_and_no_store(self):
        status, headers, body = self.get("/api/traces/newer.jsonl")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/x-ndjson; charset=utf-8")
        self.assertIn(b"session_end", body)

    def test_path_traversal_and_symlinks_are_not_read(self):
        link = self.trace_dir / "link.jsonl"
        try:
            link.symlink_to(self.trace_dir / "newer.jsonl")
        except (OSError, NotImplementedError):
            link = None
        for path in [
            "/api/traces/../older.jsonl",
            "/api/traces/%2e%2e%2Folder.jsonl",
            "/api/traces/%2Fetc%2Fpasswd.jsonl",
            "/api/traces/notes.txt",
            "/api/traces/",
        ]:
            self.assert_not_found(path)
        if link is not None:
            self.assert_not_found("/api/traces/link.jsonl")

    def test_pages_and_missing_result_route(self):
        status, headers, _body = self.get("/session-trace")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/html")
        status, headers, _body = self.get("/session_trace.js")
        self.assertEqual(status, 200)
        self.assertTrue(headers["Content-Type"].startswith("text/javascript"))
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assert_not_found("/result.json")

    def test_only_get_is_supported_for_api(self):
        with self.assertRaises(HTTPError) as context:
            urlopen(Request(self.base + "/api/traces", method="POST"))
        self.assertEqual(context.exception.code, 405)


if __name__ == "__main__":
    unittest.main()

