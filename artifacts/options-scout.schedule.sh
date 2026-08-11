#!/bin/sh
# DISABLED TEST-SAFE. Do not activate before unattended OAuth is demonstrated.
# enabled_tools=[]; generated default remains fail-closed until reviewed policy/capture proof.
set -eu
ROOT='/Users/marcboyer/Documents/Codex/2026-08-11/u'
export OPTIONS_SCOUT_ROOT="$ROOT"
cd "$ROOT"
CONSOLE="$ROOT/.venv/bin/options-scout"
PYTHON="$ROOT/.venv/bin/python"
LOCK="$ROOT/artifacts/options-scout.schedule.lock"
# DISABLED cadence templates (not installed):
# 05:30 deep-premarket: --deep
# regular +15m: --regular (local XNYS gate starts 15 minutes after open)
# 16:30 after-close: --deep
# Saturday 10:00 weekend-refresh: --deep
# 5m-active-candidate intentionally has no automatic template.
MODE="${1:-}"
if [ "$MODE" = "--regular" ]; then
  "$PYTHON" -c 'from datetime import UTC,datetime,timedelta; from options_scout.calendar import session_status; status=session_status(datetime.now(UTC)); opened=datetime.fromisoformat(str(status.get("open"))); raise SystemExit(0 if status.get("regular") is True and datetime.now(UTC).astimezone(opened.tzinfo) >= opened + timedelta(minutes=15) else 1)' || exit 0
  export OPTIONS_SCOUT_SCHEDULE_MODE="LIGHTWEIGHT_REGULAR_15M"
  MODE="--run"
elif [ "$MODE" = "--deep" ]; then
  OPTIONS_SCOUT_SCHEDULE_MODE="$("$PYTHON" -c 'from datetime import UTC,datetime; from options_scout.calendar import NY,session_status; now=datetime.now(UTC); local=now.astimezone(NY); status=session_status(now); kind=""; kind="DEEP_PREMARKET" if status.get("available") is True and status.get("session") == "PRE_OR_AFTER" and local.hour < 9 else kind; kind="DEEP_AFTER_CLOSE" if status.get("available") is True and status.get("session") == "PRE_OR_AFTER" and local.hour >= 16 else kind; kind="WEEKEND_CATALYST_REFRESH" if local.weekday() == 5 else kind; print(kind); raise SystemExit(0 if kind else 1)')" || exit 0
  export OPTIONS_SCOUT_SCHEDULE_MODE
  MODE="--run"
fi
if [ "$MODE" = "--test" ]; then
  "$PYTHON" -c 'import json,subprocess,sys; health=subprocess.run([sys.argv[1],"source-health","--json"],capture_output=True,text=True,timeout=1200); data=json.loads(health.stdout); status=data.get("allowlist_status"); assert health.returncode == 0 and status in {"EMPTY_AND_FAIL_CLOSED","REVIEWED_READ_ONLY_CAPTURE_POLICY_READY"}; assert (status == "EMPTY_AND_FAIL_CLOSED" and data.get("decision") == "DATA_INSUFFICIENT") or (status == "REVIEWED_READ_ONLY_CAPTURE_POLICY_READY" and data.get("codex_profile",{}).get("status") == "EXACT_MATCH")' "$CONSOLE"
elif [ "$MODE" = "--run" ]; then
  mkdir "$LOCK" 2>/dev/null || exit 0
  trap 'rmdir "$LOCK"' EXIT
  "$PYTHON" -c 'import json,os,pathlib,subprocess,sys; health=subprocess.run([sys.argv[1],"source-health","--json"],capture_output=True,text=True,timeout=1200); data=json.loads(health.stdout); assert health.returncode == 0 and data["allowlist_status"] == "REVIEWED_READ_ONLY_CAPTURE_POLICY_READY" and data["codex_profile"]["status"] == "EXACT_MATCH"; runner=pathlib.Path(os.environ.get("OPTIONS_SCOUT_CODEX_RUNNER", "")); assert runner.is_absolute() and runner.is_file() and os.access(runner,os.X_OK); prompt=pathlib.Path(sys.argv[2]).resolve(); root=pathlib.Path(sys.argv[3]).resolve(); assert prompt.parent == root / "artifacts"; subprocess.run([str(runner),str(prompt)],cwd=root,timeout=1200,check=True)' "$CONSOLE" "$ROOT/artifacts/options-scout.live-orchestration.md" "$ROOT"
else
  exit 64
fi
