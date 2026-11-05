import os
import subprocess
import textwrap
from pathlib import Path

import pytest


@pytest.mark.parametrize("ready_after, succeeds", [(2, True), (99, False)])
def test_deploy_waits_for_worker_startup_but_fails_when_unresponsive(
    tmp_path: Path, ready_after: int, succeeds: bool
) -> None:
    workflow = (Path(__file__).parents[3] / ".github/workflows/deploy.yml").read_text()
    start = workflow.index("          for attempt in $(seq 1 12); do")
    end = workflow.index("          migration_started=2", start)
    check = textwrap.dedent(workflow[start:end])
    for command in ("systemctl", "curl", "sleep"):
        mock = tmp_path / command
        mock.write_text("#!/bin/sh\nexit 0\n")
        mock.chmod(0o755)
    runuser = tmp_path / "runuser"
    runuser.write_text(
        '#!/bin/sh\nn=$(cat "$PING_COUNTER" 2>/dev/null || echo 0)\n'
        'n=$((n+1))\necho "$n" > "$PING_COUNTER"\n'
        '[ "$n" -ge "$READY_AFTER" ] || exit 69\nprintf "pong\\n"\n'
    )
    runuser.chmod(0o755)
    counter = tmp_path / "attempts"
    result = subprocess.run(
        ["bash", "-e", "-c", check],
        env={
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "PING_COUNTER": str(counter),
            "READY_AFTER": str(ready_after),
            "MEDIA_MODE": "full",
            "core_services": "core",
            "media_services": "media",
        },
        capture_output=True,
        timeout=5,
    )
    assert (result.returncode == 0) is succeeds
    assert int(counter.read_text()) == (2 if succeeds else 3)
