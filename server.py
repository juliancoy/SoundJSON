#!/usr/bin/env python3
import cgi
import errno
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile 
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from zipfile import BadZipFile, ZipFile


ROOT_DIR = Path(__file__).resolve().parent
SCRIPT_CANDIDATES = [
    ROOT_DIR / "soundjson.py",
    ROOT_DIR / "SoundJSON" / "sound_json.py",
]


def find_converter_script() -> Path:
    for candidate in SCRIPT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find a converter script. Checked: "
        + ", ".join(str(p) for p in SCRIPT_CANDIDATES)
    )


def is_within_directory(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def safe_extract_zip(zip_path: Path, destination: Path) -> None:
    with ZipFile(zip_path) as archive:
        for member in archive.infolist():
            member_path = destination / member.filename
            if not is_within_directory(member_path, destination):
                raise ValueError("Zip contains unsafe path traversal entries")
        archive.extractall(destination)


def run_converter(script_path: Path, input_path: Path, working_dir: Path) -> str:
    result = subprocess.run(
        [sys.executable, str(script_path), str(input_path)],
        cwd=str(working_dir),
        capture_output=True,
        text=True,
    )

    log_lines = [
        f"command: {sys.executable} {script_path} {input_path}",
        f"cwd: {working_dir}",
        f"exit_code: {result.returncode}",
    ]
    if result.stdout:
        log_lines.append("stdout:")
        log_lines.append(result.stdout.rstrip())
    if result.stderr:
        log_lines.append("stderr:")
        log_lines.append(result.stderr.rstrip())
    log_output = "\n".join(log_lines)

    if result.returncode != 0:
        raise ConverterExecutionError(
            command=f"{sys.executable} {script_path} {input_path}",
            cwd=str(working_dir),
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            log_output=log_output,
        )
    return log_output


class ConverterExecutionError(RuntimeError):
    def __init__(
        self,
        *,
        command: str,
        cwd: str,
        exit_code: int,
        stdout: str,
        stderr: str,
        log_output: str,
    ) -> None:
        super().__init__(f"Converter failed.\n{log_output}")
        self.command = command
        self.cwd = cwd
        self.exit_code = exit_code
        self.stdout = stdout or ""
        self.stderr = stderr or ""
        self.log_output = log_output


def collect_sf2_output(uploaded_file: Path) -> dict:
    output_dir = uploaded_file.with_suffix("")
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Expected output directory not found: {output_dir}")

    merged = {}
    for json_file in sorted(output_dir.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
            if isinstance(loaded, dict):
                merged.update(loaded)
    if not merged:
        raise FileNotFoundError("No JSON output produced for SF2 input")
    return merged


def collect_sfz_output(extracted_dir: Path) -> dict:
    sfz_files = sorted(extracted_dir.rglob("*.sfz"))
    if not sfz_files:
        raise FileNotFoundError("Zip archive must contain at least one .sfz file")

    merged = {}
    for sfz_file in sfz_files:
        json_file = sfz_file.with_suffix(".json")
        if not json_file.is_file():
            continue
        with open(json_file, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
            if isinstance(loaded, dict):
                merged.update(loaded)
    if not merged:
        raise FileNotFoundError("No JSON output produced for SFZ input")
    return merged


class SoundJSONHandler(BaseHTTPRequestHandler):
    server_version = "SoundJSONUploadServer/1.0"
    HEALTH_PATHS = {"/health", "/soundjson/health"}
    CONVERT_PATHS = {"/convert", "/soundjson/convert"}
    MAX_LOG_CHARS = 200000

    def _is_connection_lapse(self, exc: BaseException) -> bool:
        if isinstance(exc, (BrokenPipeError, ConnectionResetError, TimeoutError)):
            return True
        if isinstance(exc, OSError) and exc.errno in {
            errno.EPIPE,
            errno.ECONNRESET,
            errno.ETIMEDOUT,
            errno.ECONNABORTED,
        }:
            return True
        return False

    def _send_json(self, status: int, payload: dict) -> bool:
        body = json.dumps(payload, indent=2).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return True
        except Exception as exc:
            if self._is_connection_lapse(exc):
                logging.info(
                    "Client disconnected while sending JSON response: status=%s path=%s client=%s",
                    status,
                    self.path,
                    self.client_address[0] if self.client_address else "unknown",
                )
                return False
            raise

    def _classify_converter_failure(self, exc: ConverterExecutionError, extension: str) -> tuple[int, str, str]:
        stderr_text = exc.stderr
        if (
            extension == ".sf2"
            and "TypeError: object of type 'NoneType' has no len()" in stderr_text
            and "sf2utils" in stderr_text
        ):
            return (
                int(HTTPStatus.UNPROCESSABLE_ENTITY),
                "unsupported_or_corrupt_sf2",
                "The SF2 file appears corrupted or uses unsupported structure (missing PDTA data).",
            )
        if extension == ".sf2" and "corrupted but salvageable file" in stderr_text:
            return (
                int(HTTPStatus.UNPROCESSABLE_ENTITY),
                "corrupt_sf2",
                "The SF2 file appears corrupted and could not be converted.",
            )
        return (
            int(HTTPStatus.INTERNAL_SERVER_ERROR),
            "converter_failed",
            "The converter failed while processing the uploaded file.",
        )

    def log_message(self, format: str, *args) -> None:
        logging.info(
            "%s - %s",
            self.client_address[0] if self.client_address else "unknown",
            format % args,
        )

    def do_GET(self) -> None:
        request_path = urlparse(self.path).path
        if request_path in self.HEALTH_PATHS:
            self._send_json(HTTPStatus.OK, {"ok": True})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:
        request_id = uuid.uuid4().hex[:8]
        try:
            parsed_url = urlparse(self.path)
            request_path = parsed_url.path
            query = parse_qs(parsed_url.query)
            include_logs = query.get("logs", ["0"])[0].lower() in {"1", "true", "yes"}
            logging.info(
                "[%s] Incoming request: method=POST path=%s include_logs=%s client=%s",
                request_id,
                request_path,
                include_logs,
                self.client_address[0] if self.client_address else "unknown",
            )
            if request_path not in self.CONVERT_PATHS:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return

            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "Expected multipart/form-data with a 'file' field"},
                )
                return

            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": content_type,
                },
            )

            if "file" not in form:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Missing 'file' field"})
                return

            file_item = form["file"]
            if isinstance(file_item, list):
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "Please upload exactly one file in the 'file' field"},
                )
                return
            if not getattr(file_item, "filename", None):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Uploaded file has no name"})
                return

            original_name = os.path.basename(file_item.filename)
            extension = Path(original_name).suffix.lower()
            if extension not in {".sf2", ".zip"}:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "Only .sf2 files or .zip files (containing .sfz) are supported"},
                )
                return

            try:
                converter_script = find_converter_script()
            except FileNotFoundError as exc:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
                return

            with tempfile.TemporaryDirectory(prefix="soundjson-upload-") as temp_dir_str:
                temp_dir = Path(temp_dir_str)
                upload_path = temp_dir / original_name
                conversion_logs = []
                logging.info(
                    "[%s] Saved upload metadata: filename=%s extension=%s temp_dir=%s",
                    request_id,
                    original_name,
                    extension,
                    temp_dir,
                )

                with open(upload_path, "wb") as handle:
                    shutil.copyfileobj(file_item.file, handle)

                try:
                    if extension == ".sf2":
                        conversion_logs.append(run_converter(converter_script, upload_path, temp_dir))
                        result_json = collect_sf2_output(upload_path)
                    else:
                        try:
                            safe_extract_zip(upload_path, temp_dir)
                        except (BadZipFile, ValueError) as exc:
                            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                            return

                        sfz_files = sorted(temp_dir.rglob("*.sfz"))
                        if not sfz_files:
                            self._send_json(
                                HTTPStatus.BAD_REQUEST,
                                {"error": "Zip archive must include at least one .sfz file"},
                            )
                            return

                        for sfz_file in sfz_files:
                            conversion_logs.append(run_converter(converter_script, sfz_file, sfz_file.parent))
                        result_json = collect_sfz_output(temp_dir)
                except ConverterExecutionError as exc:
                    tb_text = traceback.format_exc()
                    logs_text = "\n\n".join(conversion_logs + [exc.log_output])
                    if len(logs_text) > self.MAX_LOG_CHARS:
                        logs_text = logs_text[-self.MAX_LOG_CHARS :]
                    status, error_code, message = self._classify_converter_failure(exc, extension)
                    logging.error(
                        "[%s] Converter failed: filename=%s extension=%s script=%s temp_dir=%s status=%s code=%s\n%s",
                        request_id,
                        original_name,
                        extension,
                        converter_script,
                        temp_dir,
                        status,
                        error_code,
                        tb_text,
                    )
                    error_payload = {
                        "error": message,
                        "code": error_code,
                        "request_id": request_id,
                    }
                    if include_logs:
                        error_payload["logs"] = logs_text
                        error_payload["traceback"] = tb_text
                        error_payload["converter"] = {
                            "command": exc.command,
                            "cwd": exc.cwd,
                            "exit_code": exc.exit_code,
                        }
                    self._send_json(status, error_payload)
                    return
                except Exception as exc:
                    tb_text = traceback.format_exc()
                    logs_text = "\n\n".join(conversion_logs)
                    if len(logs_text) > self.MAX_LOG_CHARS:
                        logs_text = logs_text[-self.MAX_LOG_CHARS :]
                    logging.error(
                        "[%s] Conversion failed: filename=%s extension=%s script=%s temp_dir=%s\n%s",
                        request_id,
                        original_name,
                        extension,
                        converter_script,
                        temp_dir,
                        tb_text,
                    )
                    error_payload = {
                        "error": "Conversion failed",
                        "detail": str(exc),
                        "code": "conversion_failed",
                        "request_id": request_id,
                    }
                    if include_logs:
                        error_payload["logs"] = logs_text
                        error_payload["traceback"] = tb_text
                    self._send_json(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        error_payload,
                    )
                    return

                logs_text = "\n\n".join(conversion_logs)
                if len(logs_text) > self.MAX_LOG_CHARS:
                    logs_text = logs_text[-self.MAX_LOG_CHARS :]

                if include_logs:
                    self._send_json(
                        HTTPStatus.OK,
                        {"result": result_json, "logs": logs_text, "request_id": request_id},
                    )
                    return

                body = json.dumps(result_json, indent=2).encode("utf-8")
                try:
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Disposition", 'attachment; filename="SoundJson.json"')
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except Exception as exc:
                    if self._is_connection_lapse(exc):
                        logging.info(
                            "[%s] Client disconnected while sending result body: path=%s client=%s",
                            request_id,
                            self.path,
                            self.client_address[0] if self.client_address else "unknown",
                        )
                        self.close_connection = True
                        return
                    raise
        except Exception as exc:
            if self._is_connection_lapse(exc):
                logging.info(
                    "[%s] Request terminated due to client connection lapse: path=%s client=%s",
                    request_id,
                    self.path,
                    self.client_address[0] if self.client_address else "unknown",
                )
                self.close_connection = True
                return
            raise


def main() -> None:
    host = os.environ.get("SOUNDJSON_HOST", "0.0.0.0")
    port = int(os.environ.get("SOUNDJSON_PORT", "8000"))
    log_level = os.environ.get("SOUNDJSON_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    server = ThreadingHTTPServer((host, port), SoundJSONHandler)
    print(f"Serving SoundJSON upload API on http://{host}:{port}")
    print("POST /convert with multipart/form-data field name: file")
    server.serve_forever()


if __name__ == "__main__":
    main()
