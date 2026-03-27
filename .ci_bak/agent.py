#!/usr/bin/env python3
"""CI Runner Agent: webhook server, resource-aware scheduler, GitHub status reporting.

Usage:
    # Run jobs locally (or dispatch to remote agents)
    python .ci/agent.py run
    python .ci/agent.py run --branch master --job nvidia_gpu --dry-run

    # Start webhook server (auto-detects platform)
    python .ci/agent.py serve --port 8080
"""

import argparse
import collections
import hashlib
import hmac
import json
import os
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

try:
    import yaml
except ImportError:
    print(
        "error: pyyaml is required. Install with: pip install pyyaml", file=sys.stderr
    )
    sys.exit(1)

import ci_resource as res
import github_status as gh
import run

# Maximum POST body size (1 MB) to prevent memory exhaustion
MAX_CONTENT_LENGTH = 1 * 1024 * 1024

# Job states
STATE_QUEUED = "queued"
STATE_RUNNING = "running"
STATE_PENDING = "pending"
STATE_SUCCESS = "success"
STATE_FAILURE = "failure"
STATE_ERROR = "error"

# urllib helpers (module-level for easier mocking in tests)
urllib_request = urllib.request.Request
urllib_urlopen = urllib.request.urlopen


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class JobRequest:
    """Describes a CI job to be executed."""

    def __init__(self, job_name, branch, commit_sha, config, image_tag=None, results_dir=None):
        self.job_id = str(uuid.uuid4())[:8]
        self.job_name = job_name
        self.branch = branch
        self.commit_sha = commit_sha
        self.config = config
        self.image_tag = image_tag
        self.results_dir = results_dir or Path("ci-results")
        self.created_at = datetime.now().isoformat()

        job = config["jobs"][job_name]
        self.platform = job.get("platform", "nvidia")

    def to_dict(self):
        return {
            "job_id": self.job_id,
            "job_name": self.job_name,
            "branch": self.branch,
            "commit_sha": self.commit_sha,
            "platform": self.platform,
            "created_at": self.created_at,
        }


class JobResult:
    """Outcome of a completed job."""

    def __init__(self, job_id, job_name, commit_sha, returncode, results_dir, duration):
        self.job_id = job_id
        self.job_name = job_name
        self.commit_sha = commit_sha
        self.returncode = returncode
        self.results_dir = results_dir
        self.duration = duration

        self.state = STATE_SUCCESS if returncode == 0 else STATE_FAILURE

    def to_dict(self):
        return {
            "job_id": self.job_id,
            "job_name": self.job_name,
            "commit_sha": self.commit_sha,
            "state": self.state,
            "returncode": self.returncode,
            "results_dir": str(self.results_dir),
            "duration_seconds": round(self.duration, 1),
        }


# ---------------------------------------------------------------------------
# Job selection and routing
# ---------------------------------------------------------------------------


def select_jobs(config, platform=None, job_name=None):
    """Return list of job names to run."""
    jobs = config.get("jobs", {})

    if job_name:
        if job_name not in jobs:
            raise ValueError(f"job {job_name!r} not in config")

        return [job_name]

    if platform:
        return [
            name for name, job in jobs.items() if job.get("platform") == platform
        ]

    return list(jobs.keys())



# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class Scheduler:
    """Resource-aware job scheduler with dynamic parallelism."""

    def __init__(
        self,
        config,
        platform,
        resource_pool,
        results_dir=None,
        max_workers=4,
        no_status=False,
        dry_run=False,
    ):
        self._config = config
        self._platform = platform
        self._resource_pool = resource_pool
        self._results_dir = results_dir or Path("ci-results")
        self._no_status = no_status
        self._dry_run = dry_run
        self._queue = collections.deque()
        self._jobs: dict[str, dict] = {}  # job_id -> {request, result, state, gpu_ids}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        self._done_event = threading.Event()

        # GitHub config
        github_cfg = config.get("github", {})
        self._status_prefix = github_cfg.get("status_context_prefix", "ci/infiniops")
        repo = config.get("repo", {})
        repo_url = repo.get("url", "")
        self._owner, self._repo = gh.parse_repo_url(repo_url)

    def submit(self, job_request):
        """Add a job to the queue and attempt to schedule it.

        Returns the job_id.
        """
        with self._lock:
            self._jobs[job_request.job_id] = {
                "request": job_request,
                "result": None,
                "state": STATE_QUEUED,
                "gpu_ids": [],
            }
            self._queue.append(job_request)

        self._try_schedule()
        return job_request.job_id

    def get_job(self, job_id):
        """Get job info by ID."""
        with self._lock:
            entry = self._jobs.get(job_id)

            if not entry:
                return None

            info = entry["request"].to_dict()
            info["state"] = entry["state"]

            if entry["result"]:
                info.update(entry["result"].to_dict())

            return info

    def get_status(self):
        """Return scheduler status for the /status endpoint."""
        with self._lock:
            queued = [
                self._jobs[r.job_id]["request"].to_dict()
                for r in self._queue
            ]
            running = []
            completed = []

            for entry in self._jobs.values():
                state = entry["state"]

                if state == STATE_RUNNING:
                    running.append({**entry["request"].to_dict(), "gpu_ids": entry["gpu_ids"]})
                elif state in (STATE_SUCCESS, STATE_FAILURE):
                    completed.append(entry["result"].to_dict())

        return {
            "queued": queued,
            "running": running,
            "completed": completed[-20:],  # Last 20
            "resources": self._resource_pool.get_status(),
        }

    def wait_all(self):
        """Block until all submitted jobs are done. Returns list of JobResult."""
        while True:
            with self._lock:
                pending = any(
                    e["state"] in (STATE_QUEUED, STATE_RUNNING) for e in self._jobs.values()
                )

            if not pending:
                break

            self._done_event.wait(timeout=2.0)
            self._done_event.clear()

        with self._lock:
            return [
                e["result"]
                for e in self._jobs.values()
                if e["result"] is not None
            ]

    def _try_schedule(self):
        """Try to run queued jobs that have enough resources.

        Resource allocation and job submission are split: allocation decisions
        are made under the lock, but executor.submit() happens outside to
        prevent deadlock when the thread pool is saturated.
        """
        to_launch = []  # [(req, gpu_ids), ...]

        with self._lock:
            remaining = collections.deque()

            while self._queue:
                req = self._queue.popleft()
                job_cfg = self._config["jobs"].get(req.job_name, {})
                gpu_count = res.parse_gpu_requirement(job_cfg)
                memory_mb = res.parse_memory_requirement(job_cfg)

                if self._dry_run:
                    # In dry-run mode, skip resource checks
                    gpu_ids, ok = [], True
                else:
                    gpu_ids, ok = self._resource_pool.allocate(gpu_count, memory_mb)

                if ok:
                    self._jobs[req.job_id]["state"] = STATE_RUNNING
                    self._jobs[req.job_id]["gpu_ids"] = gpu_ids
                    to_launch.append((req, gpu_ids))
                else:
                    remaining.append(req)

            self._queue = remaining

        # Submit outside the lock to avoid deadlock with ThreadPoolExecutor
        for req, gpu_ids in to_launch:
            self._executor.submit(self._run_job, req, gpu_ids)

    def _run_job(self, req, gpu_ids):
        """Execute a single job in a worker thread.

        Wrapped in try/finally to guarantee GPU resources are always released
        and job state is updated even on unexpected exceptions.
        """
        context = gh.build_status_context(self._status_prefix, req.job_name)
        result = None

        try:
            # Post pending status
            if not self._no_status:
                gh.post_commit_status(
                    self._owner,
                    self._repo,
                    req.commit_sha,
                    STATE_PENDING,
                    context,
                    f"Running {req.job_name}...",
                )

            job_cfg = self._config["jobs"][req.job_name]
            all_stages = job_cfg.get("stages", [])
            repo_url = self._config.get("repo", {}).get("url", "")
            commit_short = req.commit_sha[:7] if len(req.commit_sha) > 7 else req.commit_sha
            results_dir = run.build_results_dir(
                req.results_dir, req.platform, all_stages, commit_short
            )

            gpu_id_str = ",".join(str(g) for g in gpu_ids) if gpu_ids else None
            docker_args = run.build_docker_args(
                self._config,
                req.job_name,
                repo_url,
                req.branch,
                all_stages,
                "/workspace",
                req.image_tag,
                gpu_id_override=gpu_id_str,
                results_dir=results_dir,
            )

            start = time.monotonic()

            if self._dry_run:
                print(f"[dry-run] {req.job_name}: {shlex.join(docker_args)}")
                returncode = 0
            else:
                results_dir.mkdir(parents=True, exist_ok=True)
                proc = subprocess.run(docker_args)
                returncode = proc.returncode

            duration = time.monotonic() - start

            result = JobResult(
                job_id=req.job_id,
                job_name=req.job_name,
                commit_sha=req.commit_sha,
                returncode=returncode,
                results_dir=results_dir,
                duration=duration,
            )

            # Post final status
            if not self._no_status:
                gh.post_commit_status(
                    self._owner,
                    self._repo,
                    req.commit_sha,
                    result.state,
                    context,
                    f"{req.job_name}: {result.state} in {duration:.0f}s",
                )
        except Exception as e:
            print(f"error: job {req.job_name} failed with exception: {e}", file=sys.stderr)

            if result is None:
                result = JobResult(
                    job_id=req.job_id,
                    job_name=req.job_name,
                    commit_sha=req.commit_sha,
                    returncode=-1,
                    results_dir=req.results_dir,
                    duration=0,
                )

            if not self._no_status:
                gh.post_commit_status(
                    self._owner,
                    self._repo,
                    req.commit_sha,
                    STATE_ERROR,
                    context,
                    f"{req.job_name}: internal error",
                )
        finally:
            # Always release resources and update state
            self._resource_pool.release(gpu_ids)

            with self._lock:
                self._jobs[req.job_id]["result"] = result
                self._jobs[req.job_id]["state"] = result.state if result else STATE_FAILURE

            self._done_event.set()
            self._try_schedule()

        return result


# ---------------------------------------------------------------------------
# Webhook server
# ---------------------------------------------------------------------------


def verify_signature(secret, body, signature_header):
    """Verify GitHub webhook HMAC-SHA256 signature."""
    if not signature_header:
        return False

    expected = "sha256=" + hmac.new(
        secret.encode("utf-8"), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature_header)


def _verify_api_token(handler):
    """Check Bearer token for /api/run authentication.

    Returns True if authenticated, False (and sends 401) if not.
    When no api_token is configured on the server, all requests are allowed.
    """
    api_token = getattr(handler.server, "api_token", None)

    if not api_token:
        return True

    auth_header = handler.headers.get("Authorization", "")

    if auth_header == f"Bearer {api_token}":
        return True

    handler._respond_json(401, {"error": "unauthorized"})
    return False


class WebhookHandler(BaseHTTPRequestHandler):
    """HTTP handler for GitHub webhooks and API endpoints."""

    def log_message(self, format, *args):
        print(f"[agent] {args[0]}", file=sys.stderr)

    def do_GET(self):
        if self.path == "/health":
            self._respond_json(200, {"status": "ok", "platform": self.server.platform})
        elif self.path == "/status":
            status = self.server.scheduler.get_status()
            self._respond_json(200, status)
        elif self.path.startswith("/api/job/"):
            self._handle_api_job()
        else:
            self._respond_json(404, {"error": "not found"})

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))

        if content_length > MAX_CONTENT_LENGTH:
            self._respond_json(413, {"error": "payload too large"})
            return

        body = self.rfile.read(content_length)

        if self.path == "/webhook":
            self._handle_webhook(body)
        elif self.path == "/api/run":
            self._handle_api_run(body)
        else:
            self._respond_json(404, {"error": "not found"})

    def _handle_webhook(self, body):
        # Verify signature if secret is configured
        if self.server.webhook_secret:
            sig = self.headers.get("X-Hub-Signature-256", "")

            if not verify_signature(self.server.webhook_secret, body, sig):
                self._respond_json(401, {"error": "invalid signature"})
                return

        event_type = self.headers.get("X-GitHub-Event", "")

        if event_type == "ping":
            self._respond_json(200, {"msg": "pong"})
            return

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self._respond_json(400, {"error": "invalid JSON"})
            return

        if event_type == "push":
            branch, sha = self._parse_push(payload)
        elif event_type == "pull_request":
            action = payload.get("action", "")

            if action not in ("opened", "synchronize"):
                self._respond_json(200, {"msg": f"ignored PR action: {action}"})
                return

            branch, sha = self._parse_pull_request(payload)
        else:
            self._respond_json(200, {"msg": f"ignored event: {event_type}"})
            return

        if not branch or not sha:
            self._respond_json(400, {"error": "could not extract branch/sha"})
            return

        job_ids = self._submit_jobs(branch, sha)
        self._respond_json(200, {"accepted": True, "job_ids": job_ids})

    def _handle_api_run(self, body):
        """Handle /api/run: remote job trigger (requires Bearer token auth)."""
        if not _verify_api_token(self):
            return

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self._respond_json(400, {"error": "invalid JSON"})
            return

        branch = payload.get("branch", "")
        sha = payload.get("commit_sha", "")
        job_name = payload.get("job")
        image_tag = payload.get("image_tag")

        if not branch:
            self._respond_json(400, {"error": "branch is required"})
            return

        if not sha:
            sha = run.get_git_commit()

        job_ids = self._submit_jobs(branch, sha, job_name=job_name, image_tag=image_tag)
        self._respond_json(200, {"accepted": True, "job_ids": job_ids})

    def _handle_api_job(self):
        """Handle GET /api/job/{id}."""
        parts = self.path.split("/")

        if len(parts) < 4:
            self._respond_json(400, {"error": "missing job_id"})
            return

        job_id = parts[3]
        info = self.server.scheduler.get_job(job_id)

        if info is None:
            self._respond_json(404, {"error": f"job {job_id} not found"})
        else:
            self._respond_json(200, info)

    def _parse_push(self, payload):
        branch = payload.get("ref", "").removeprefix("refs/heads/")
        sha = payload.get("after", "")
        return branch, sha

    def _parse_pull_request(self, payload):
        pr = payload.get("pull_request", {})
        head = pr.get("head", {})
        branch = head.get("ref", "")
        sha = head.get("sha", "")
        return branch, sha

    def _submit_jobs(self, branch, sha, job_name=None, image_tag=None):
        config = self.server.config
        job_names = select_jobs(config, platform=self.server.platform, job_name=job_name)
        job_ids = []

        for name in job_names:
            req = JobRequest(
                job_name=name,
                branch=branch,
                commit_sha=sha,
                config=config,
                image_tag=image_tag,
                results_dir=self.server.results_dir,
            )
            jid = self.server.scheduler.submit(req)
            job_ids.append(jid)

        return job_ids

    def _respond_json(self, status_code, data):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class AgentServer(HTTPServer):
    """HTTP server with scheduler and config context."""

    def __init__(
        self,
        host,
        port,
        config,
        scheduler,
        platform,
        webhook_secret=None,
        api_token=None,
        results_dir=None,
    ):
        super().__init__((host, port), WebhookHandler)
        self.config = config
        self.scheduler = scheduler
        self.platform = platform
        self.webhook_secret = webhook_secret
        self.api_token = api_token
        self.results_dir = results_dir or Path("ci-results")


# ---------------------------------------------------------------------------
# Remote job dispatch (for CLI triggering remote agents)
# ---------------------------------------------------------------------------


def dispatch_remote_job(agent_url, job_name, branch, commit_sha, image_tag=None, api_token=None):
    """Send a job to a remote agent via HTTP API. Returns job_id or None."""
    url = f"{agent_url.rstrip('/')}/api/run"
    body = {
        "branch": branch,
        "commit_sha": commit_sha,
        "job": job_name,
    }

    if image_tag:
        body["image_tag"] = image_tag

    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    req = urllib_request(url, data=data, headers=headers, method="POST")

    try:
        with urllib_urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            job_ids = result.get("job_ids", [])
            return job_ids[0] if job_ids else None
    except Exception as e:
        print(f"error: failed to dispatch to {agent_url}: {e}", file=sys.stderr)
        return None


def poll_remote_job(agent_url, job_id, interval=5.0, timeout=7200):
    """Poll a remote agent for job completion. Returns final state dict or None."""
    url = f"{agent_url.rstrip('/')}/api/job/{job_id}"
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        try:
            req = urllib_request(url)

            with urllib_urlopen(req, timeout=10) as resp:
                info = json.loads(resp.read())

            state = info.get("state", "")

            if state in (STATE_SUCCESS, STATE_FAILURE):
                return info
        except Exception:
            pass

        time.sleep(interval)

    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_run(args):
    """Handle 'run' subcommand: dispatch jobs to platform agents via HTTP."""
    config = run.load_config(args.config)
    agents = config.get("agents", {})
    branch = args.branch or config.get("repo", {}).get("branch", "master")
    commit_sha = args.commit or run.get_git_commit(short=False)

    # Determine which jobs to run
    try:
        job_names = select_jobs(config, platform=args.platform, job_name=args.job)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    if not job_names:
        print("error: no matching jobs found", file=sys.stderr)
        sys.exit(1)

    # Resolve agent URL for each job
    jobs_to_dispatch = []  # [(name, agent_url)]

    for name in job_names:
        job = config.get("jobs", {}).get(name, {})
        platform = job.get("platform", "")
        agent_url = agents.get(platform, {}).get("url", "")

        if not agent_url:
            print(f"error: no agent URL configured for platform {platform!r} (job {name})", file=sys.stderr)
            sys.exit(1)

        jobs_to_dispatch.append((name, agent_url))

    api_token = os.environ.get("AGENT_API_TOKEN", "")
    results = []

    if args.dry_run:
        for name, agent_url in jobs_to_dispatch:
            platform, _, job = name.partition("_")
            print(f"[dry-run] dispatch {platform} {job} job to {agent_url}")
    else:
        # Dispatch all jobs, then poll concurrently.
        dispatched = []  # [(name, agent_url, job_id)]

        for name, agent_url in jobs_to_dispatch:
            platform, _, job = name.partition("_")
            print(
                f"==> dispatching {platform} {job} job to {agent_url}",
                file=sys.stderr,
            )
            job_id = dispatch_remote_job(
                agent_url, name, branch, commit_sha, args.image_tag,
                api_token=api_token or None,
            )

            if job_id:
                print(f"    job_id: {job_id}", file=sys.stderr)
                dispatched.append((name, agent_url, job_id))
            else:
                print(f"    failed to dispatch {name}", file=sys.stderr)
                results.append({"job_name": name, "state": "error"})

        if dispatched:
            with ThreadPoolExecutor(max_workers=len(dispatched)) as executor:
                futures = {
                    executor.submit(poll_remote_job, url, jid): (name, url, jid)
                    for name, url, jid in dispatched
                }

                for future in as_completed(futures):
                    name, _, _ = futures[future]
                    result = future.result()

                    if result:
                        state = result.get("state", "unknown")
                        duration = result.get("duration_seconds", 0)
                        tag = "PASS" if state == STATE_SUCCESS else "FAIL"
                        print(
                            f"<== {tag}  {name}  ({duration:.0f}s)",
                            file=sys.stderr,
                        )
                        results.append(result)
                    else:
                        print(f"<== TIMEOUT  {name}", file=sys.stderr)
                        results.append({"job_name": name, "state": "timeout"})

    # Summary
    print("\n========== Results ==========")
    all_ok = True

    for r in results:
        state = r.get("state", "unknown")
        name = r.get("job_name", "?")
        status = "PASS" if state == STATE_SUCCESS else "FAIL"

        if state != STATE_SUCCESS:
            all_ok = False

        duration = r.get("duration_seconds", 0)
        print(f"  {status}  {name}  ({duration:.0f}s)")

    if not all_ok:
        sys.exit(1)


def cmd_serve(args):
    """Handle 'serve' subcommand: start webhook server."""
    config = run.load_config(args.config)

    platform = res.detect_platform()

    if not platform:
        print(
            "error: could not detect platform (no nvidia-smi or ixsmi found)",
            file=sys.stderr,
        )
        sys.exit(1)

    platform_jobs = select_jobs(config, platform=platform)

    if not platform_jobs:
        print(
            f"error: platform {platform!r} detected but no jobs defined in config",
            file=sys.stderr,
        )
        sys.exit(1)

    pool = res.ResourcePool(
        platform,
        utilization_threshold=args.utilization_threshold,
    )
    scheduler = Scheduler(
        config,
        platform,
        pool,
        results_dir=args.results_dir,
    )

    webhook_secret = args.webhook_secret or os.environ.get("WEBHOOK_SECRET", "")
    api_token = args.api_token or os.environ.get("AGENT_API_TOKEN", "")

    if not webhook_secret:
        print(
            "WARNING: No webhook secret configured. Webhook endpoint accepts "
            "unsigned requests. Set --webhook-secret or WEBHOOK_SECRET for production.",
            file=sys.stderr,
        )

    if not api_token:
        print(
            "WARNING: No API token configured. /api/run endpoint is unauthenticated. "
            "Set --api-token or AGENT_API_TOKEN for production.",
            file=sys.stderr,
        )

    server = AgentServer(
        args.host,
        args.port,
        config,
        scheduler,
        platform,
        webhook_secret=webhook_secret or None,
        api_token=api_token or None,
        results_dir=args.results_dir,
    )

    print(
        f"Agent serving on {args.host}:{args.port} (platform={platform})",
        file=sys.stderr,
    )
    print(f"  POST /webhook  — GitHub webhook", file=sys.stderr)
    print(f"  POST /api/run  — remote job trigger", file=sys.stderr)
    print(f"  GET  /health   — health check", file=sys.stderr)
    print(f"  GET  /status   — queue & resource status", file=sys.stderr)
    print(f"  GET  /api/job/{{id}} — job status", file=sys.stderr)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...", file=sys.stderr)
        server.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="CI Runner Agent: run jobs locally, dispatch remotely, or serve webhooks",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- run subcommand ---
    run_parser = subparsers.add_parser("run", help="Run CI jobs")
    run_parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config.yaml",
    )
    run_parser.add_argument("--branch", type=str, help="Branch to test (default: config repo.branch)")
    run_parser.add_argument("--job", type=str, help="Specific job name")
    run_parser.add_argument("--platform", type=str, help="Filter jobs by platform")
    run_parser.add_argument("--image-tag", type=str, help="Override image tag")
    run_parser.add_argument("--commit", type=str, help="Override commit SHA")
    run_parser.add_argument("--dry-run", action="store_true")

    # --- serve subcommand ---
    serve_parser = subparsers.add_parser("serve", help="Start webhook server")
    serve_parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config.yaml",
    )
    serve_parser.add_argument("--port", type=int, default=8080)
    serve_parser.add_argument("--host", type=str, default="0.0.0.0")
    serve_parser.add_argument("--webhook-secret", type=str)
    serve_parser.add_argument(
        "--api-token",
        type=str,
        help="Bearer token for /api/run authentication (or AGENT_API_TOKEN env var)",
    )
    serve_parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("ci-results"),
    )
    serve_parser.add_argument(
        "--utilization-threshold",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
