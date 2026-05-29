# ddharmon GUI Deployment (internal VM — systemd + nginx)

Step-by-step guide for deploying the ddharmon harmonization GUI to a single always-on VM
(e.g. AWS Lightsail / an internal Ubuntu host), mirroring the biomapper-ui deploy pattern.

Replace `$DEPLOY_DIR` (e.g. `/home/ubuntu/ddharmon`) and `$DOMAIN`
(e.g. `harmonize.expertintheloop.io`) throughout.

## Architecture (read first)

```
browser ──HTTPS──> nginx (:443, TLS) ──proxy──> uvicorn (127.0.0.1:8000)
                                                   └─ FastAPI: serves the built SPA + /api
                                                      └─ ddharmon pipeline (in-process):
                                                         embed (sentence-transformers/torch)
                                                         → BERTopic (UMAP/HDBSCAN)
                                                         → value sub-cluster → CDE anchor
                                                         → classify (none | sync | batch)
```

**Why a single always-on instance (NOT autoscale / scale-to-zero):**
- The job store is **in-memory** — jobs are lost on restart and invisible across processes.
- Progress is streamed over **SSE** (long-lived connections).
- The ML pipeline runs **in-process** in a background thread.

→ Run exactly **one uvicorn process, one worker**. Do not add `--workers >1`.

## Prerequisites

- Ubuntu host with **≥ 4 GB RAM** (8 GB comfortable — torch + UMAP/HDBSCAN are memory-hungry),
  ~6 GB free disk (torch + the ~420 MB embedding model + node_modules).
- **Python 3.12+** (the project requires `>=3.12`).
- **Node 20+** and npm (for the one-time frontend build).
- **nginx**, **certbot** (`python3-certbot-nginx`).
- **uv** (recommended) or pip.
- Read access to the **private** repo `Phenome-Health/ddharmon` (deploy key or PAT).

```bash
# uv (if not present)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 1. DNS Setup (do early)

Create an A record pointing `$DOMAIN` at the instance IP — propagation can take minutes to hours.

```bash
dig +short $DOMAIN     # later: should return your instance IP
```

## 2. Clone Repository

```bash
ssh ubuntu@<INSTANCE_IP>
cd ~
git clone https://github.com/Phenome-Health/ddharmon.git
cd ddharmon                                  # this is $DEPLOY_DIR
```

## 3. Python Dependencies (the full ML stack — not just `[ui]`)

The pipeline runner imports sentence-transformers + BERTopic/UMAP/HDBSCAN, so install `[all]`
(the bare `ui` extra is not enough). This pulls torch and takes several minutes.

```bash
uv venv
uv pip install -e ".[all]"
# or, without uv:  python3 -m venv .venv && .venv/bin/pip install -e ".[all]"
```

## 4. Build the Frontend (one-time; output is gitignored)

The FastAPI app mounts `ui/frontend/dist` at `/` when present. `dist/` is not in git, so build it:

```bash
cd ui/frontend
npm ci
npm run build          # produces ui/frontend/dist
cd ../..
```

## 5. CDE Catalog (server-side; NOT in git)

`data/examples/cde/` is gitignored, so a fresh clone has **no** catalog. Pick one:

- **No CDE anchoring** — run with `cdeSet = none` in the UI. Nothing to install here; skip to step 6.
- **With CDE anchoring** — get the catalog onto the server, then it loads automatically:
  - The app expects `data/examples/cde/nih_endorsed_flat.tsv` (cdeSet `endorsed`) and/or
    `data/examples/cde/all_cdes_flat.tsv` (cdeSet `full`).
  - Either copy the prebuilt flat TSVs from a machine that has them:
    ```bash
    # from your laptop:
    scp data/examples/cde/nih_endorsed_flat.tsv data/examples/cde/all_cdes_flat.tsv \
        ubuntu@<INSTANCE_IP>:$DEPLOY_DIR/data/examples/cde/
    ```
  - …or copy the source JSON and regenerate on the server (`flatten_cde_repo.py` ships in the repo):
    ```bash
    mkdir -p data/examples/cde
    # scp All-CDEs.json / NIH-endorsed-CDEs.json into data/examples/cde first, then:
    .venv/bin/python scripts/flatten_cde_repo.py \
        data/examples/cde/NIH-endorsed-CDEs.json data/examples/cde/nih_endorsed_flat.tsv
    .venv/bin/python scripts/flatten_cde_repo.py \
        data/examples/cde/All-CDEs.json data/examples/cde/all_cdes_flat.tsv
    ```

## 6. Environment File

```bash
cp deploy/.env.example .env
nano .env       # set ANTHROPIC_API_KEY only if you'll use classifyMode sync|batch
```

## 7. Warm the Embedding Model (recommended)

Pre-download `all-mpnet-base-v2` (~420 MB) into the deploy-dir cache now, so the first run isn't
slow and any download/permission issue surfaces at deploy time rather than mid-job:

```bash
HF_HOME=$DEPLOY_DIR/.cache/huggingface \
SENTENCE_TRANSFORMERS_HOME=$DEPLOY_DIR/.cache/sentence-transformers \
.venv/bin/python -c "from ddharmon.embedding.provider import SentenceTransformerProvider; SentenceTransformerProvider()"
```

## 8. systemd Service

```bash
sudo cp deploy/ddharmon.service /etc/systemd/system/
sudo sed -i "s|\$DEPLOY_DIR|$DEPLOY_DIR|g" /etc/systemd/system/ddharmon.service
# If the service user isn't `ubuntu`, also edit User=/Group= in the unit.

sudo systemctl daemon-reload
sudo systemctl enable ddharmon
sudo systemctl start ddharmon
```

### Service Management

```bash
sudo systemctl status ddharmon
sudo journalctl -u ddharmon -f       # logs (model load, job progress, errors)
sudo systemctl restart ddharmon
```

## 9. nginx Configuration

```bash
sudo cp deploy/nginx-ddharmon.conf /etc/nginx/sites-available/ddharmon.conf
sudo sed -i "s|\$DOMAIN|$DOMAIN|g" /etc/nginx/sites-available/ddharmon.conf
sudo ln -s /etc/nginx/sites-available/ddharmon.conf /etc/nginx/sites-enabled/

sudo nginx -t
sudo systemctl reload nginx
```

## 10. DNS Propagation Check + SSL

```bash
dig +short $DOMAIN                   # confirm it resolves to this host first
sudo certbot --nginx -d $DOMAIN      # adds :443 listener + HTTP->HTTPS redirect
```

## 11. (Optional but recommended) Access Gate

The app has no built-in auth. If `$DOMAIN` is reachable beyond a trusted network, enable
basic-auth **after** TLS is active (so credentials aren't sent in the clear):

```bash
sudo apt-get install -y apache2-utils
sudo htpasswd -c /etc/nginx/.ddharmon_htpasswd <user>
# uncomment the two auth_basic lines in /etc/nginx/sites-available/ddharmon.conf, then:
sudo nginx -t && sudo systemctl reload nginx
```

## 12. Verification

```bash
# App is up (internal):
curl -s http://127.0.0.1:8000/api/health
#   -> {"status":"ok","version":"1.0.0","cde":{"endorsed":true|false,"full":...},"frontendBuilt":true}

# Through nginx + TLS:
curl -s https://$DOMAIN/api/health
curl -s -o /dev/null -w "%{http_code}\n" https://$DOMAIN/      # SPA index -> 200

sudo systemctl is-active ddharmon
```

`frontendBuilt` should be `true` (step 4); each `cde.*` flag reflects which catalog file you
installed in step 5 (both `false` is fine if you only use `cdeSet = none`).

## 13. Redeploy / Update

```bash
cd $DEPLOY_DIR
git pull
uv pip install -e ".[all]"          # if deps changed
(cd ui/frontend && npm ci && npm run build)   # if frontend changed
sudo systemctl restart ddharmon     # NOTE: in-memory jobs are lost on restart
```

## Notes & Gotchas

- **In-memory jobs** are lost on every restart — fine for the single-user v1 tool; warn users
  before restarting mid-run.
- **One worker only.** Multiple uvicorn workers would split the in-memory job store and break SSE.
- **Memory:** a large `cdeSet = full` run plus a big cohort upload can spike RAM (embeddings +
  UMAP). If the service is OOM-killed (check `journalctl`/`dmesg`), size up the instance or
  prefer `cdeSet = endorsed`.
- **systemd EnvironmentFile precedence:** under systemd, `.env` values come from `EnvironmentFile`;
  when running manually for debugging, `load_dotenv()` reads `.env` instead.
- **First job is slow** if you skipped step 7 (model downloads on first embed).
- **SSE through nginx** relies on `proxy_buffering off` + a long `proxy_read_timeout` — both are
  set in `nginx-ddharmon.conf`; don't re-enable buffering.
