#!/bin/bash
set -euo pipefail
exec 9>/srv/resume-ai/deploy.lock
flock -n 9 || exit 0
cd /srv/resume-ai/source
git fetch --quiet origin main
revision=$(git rev-parse origin/main)
current=$(readlink /srv/resume-ai/current || true)
release=/srv/resume-ai/releases/$revision
if [ "$current" = "$release" ] && systemctl is-active --quiet resume-ai; then exit 0; fi
mkdir -p "$release"
git archive "$revision" | tar -x -C "$release"
/srv/resume-ai/venv/bin/pip install --quiet -r "$release/requirements-server.txt"
/srv/resume-ai/venv/bin/python -m compileall -q "$release/main.py" "$release/text_retrieval.py"
ln -sfn "$release" /srv/resume-ai/current
systemctl restart resume-ai
for attempt in $(seq 1 20); do
 if curl -fsS http://127.0.0.1:8200/health | grep -q '"has_resume_db":true'; then
  printf '%s %s\n' "$(date -Is)" "$revision" >> /srv/resume-ai/deployments.log
  exit 0
 fi
 sleep 1
done
if [ -n "$current" ]; then ln -sfn "$current" /srv/resume-ai/current; systemctl restart resume-ai; fi
exit 1
