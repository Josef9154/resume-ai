# Resume AI deployment rules
This is Joseph Rochelle's virtual resume, not CODE OS.
- Preserve verified resume facts and the approved profile photo.
- Never commit .env files, keys, backups, unrelated projects, or payment configuration.
- Production: separate resume-ai service on payments OVH VPS; loopback port 8200.
- Ollama: private localhost only, Qwen2.5 1.5B, bounded concurrency and memory.
- Server uses text_retrieval.py for low-memory source retrieval; local FAISS remains optional.
- Commit tested changes and push main to deploy. The server checks GitHub every minute.
- Check /JosephRochelle/health and a real chat response after deployment; never claim success from push alone.
- Do not modify payments services or their databases.

- Keep about_me.txt and profile_answers.json in deployment. Personal facts require user confirmation.
- Common answers must have matching source evidence. Never fabricate Eli’s age, breed, or habits.
- Preserve section headings with their details in retrieval. Zero-match questions must not retrieve arbitrary chunks.
- Exact-question caches must be bounded and keyed by source revision and model. Never cache truncated/error outputs.
