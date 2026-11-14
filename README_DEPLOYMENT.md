# OVH resume deployment
The existing payments VPS hosts the resume as an isolated service on localhost:8200.
Ollama uses qwen2.5:1.5b on localhost:11434; model memory and CPU are capped.
Server retrieval uses the resume.txt and optional about_me.txt source directly with a low-memory term index. FAISS remains available for local use.

## Updates from VS Code or Codex
1. Edit and test in this project.
2. Commit only intended files.
3. Push main to Josef9154/resume-ai (or run VS Code task `Resume: push to OVH`).
4. The server's systemd timer checks GitHub every minute, creates a versioned release, runs health checks and rolls back activation on failure.
Saving a file alone does not publish unfinished edits.

## Operations
Service: resume-ai; deployment timer: resume-ai-update.timer.
Logs: journalctl -u resume-ai; /srv/resume-ai/deployments.log.
Secrets: /etc/resume-ai.env (server only).
Payments services and databases are independent and must not be modified.

Public URL: https://api.mycodeexperience.com/JosephRochelle/
Health: https://api.mycodeexperience.com/JosephRochelle/health
Current Qwen model: qwen2.5:1.5b (CPU). Single active AI request; retry if busy.
