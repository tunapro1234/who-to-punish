"""
Download oTree session data via the REST API export endpoints.

oTree exposes two CSV download endpoints (since v6) that use the REST API key:
  - GET /api/export_wide               — all apps, all sessions
  - GET /api/export_app?app=<name>     — one app, all sessions
  - Both accept ?session_code=<code> to filter to one session

These return CSV directly. No admin login required.
"""

import os
import requests


class OTreeExporter:
    """
    Download experiment data from a running oTree server via REST API.

    Usage:
        exporter = OTreeExporter("http://localhost:8000", rest_key="test-rest-key")

        # Wide format (all apps in one CSV)
        path = exporter.export_wide(output_dir="results")

        # Per-app data
        path = exporter.export_app("ertan2009", output_dir="results")

        # Just one session
        path = exporter.export_app("ertan2009", session_code="abcd1234")
    """

    def __init__(self, server_url: str, rest_key: str = None,
                 admin_password: str = None):
        self.server_url = server_url.rstrip("/")
        self.rest_key = rest_key or os.environ.get("OTREE_REST_KEY", "test-rest-key")
        # admin_password kept for backwards compat / future fallback to admin login
        self.admin_password = admin_password or os.environ.get("OTREE_ADMIN_PASSWORD", "admin")

    def _headers(self) -> dict:
        return {"otree-rest-key": self.rest_key}

    def export_wide(self, output_dir: str = "results",
                    session_code: str = None,
                    filename: str = None) -> str:
        """
        Download the wide-format CSV (all apps in one file, one row per
        participant). If session_code is given, restricts to that session.
        """
        os.makedirs(output_dir, exist_ok=True)
        if filename is None:
            filename = f"otree_wide_{session_code}.csv" if session_code else "otree_wide.csv"
        path = os.path.join(output_dir, filename)

        url = f"{self.server_url}/api/export_wide"
        params = {}
        if session_code:
            params["session_code"] = session_code

        r = requests.get(url, headers=self._headers(), params=params)
        r.raise_for_status()

        with open(path, "wb") as f:
            f.write(r.content)
        return path

    def export_app(self, app_name: str, output_dir: str = "results",
                   session_code: str = None,
                   filename: str = None) -> str:
        """
        Download CSV for a specific oTree app. One row per player per round.
        """
        os.makedirs(output_dir, exist_ok=True)
        if filename is None:
            tag = f"_{session_code}" if session_code else ""
            filename = f"otree_{app_name}{tag}.csv"
        path = os.path.join(output_dir, filename)

        url = f"{self.server_url}/api/export_app"
        params = {"app": app_name}
        if session_code:
            params["session_code"] = session_code

        r = requests.get(url, headers=self._headers(), params=params)
        r.raise_for_status()

        with open(path, "wb") as f:
            f.write(r.content)
        return path

    def get_session_metadata(self, session_code: str) -> dict:
        """
        Get session metadata via the REST API.
        Returns participant info, payoffs, etc. Does NOT include
        form responses — use export_app() or export_wide() for that.
        """
        url = f"{self.server_url}/api/sessions/{session_code}"
        r = requests.get(url, headers=self._headers())
        r.raise_for_status()
        return r.json()
