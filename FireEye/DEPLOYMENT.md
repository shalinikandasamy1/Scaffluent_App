# FireEye Deployment Guide

## Quick Start (Docker)

```bash
cd FireEye
# Copy and edit your .env file
cp .env.example .env
# Add your OpenRouter API key to .env

# Build and start
docker compose up -d --build

# View logs
docker compose logs -f fireeye
```

The dashboard will be available at `http://<host>:8090`.

## Deploying to Tesla P4 (.172)

```bash
# From the dev machine (.153)
rsync -avz --exclude='.venv' --exclude='__pycache__' --exclude='runs' --exclude='.git' \
  FireEye/ evnchn@192.168.50.172:~/FireEye/

# SSH to .172 and start
ssh evnchn@192.168.50.172
cd ~/FireEye
docker compose up -d --build
```

## Authentication

Authentication is **optional** and controlled by the `FIREEYE_AUTH_ENABLED` environment variable.

| Variable | Default | Description |
|----------|---------|-------------|
| `FIREEYE_AUTH_ENABLED` | `false` | Set to `true` to require login |
| `FIREEYE_USERS` | `admin:fireeye` | Comma-separated `user:pass` pairs |
| `FIREEYE_STORAGE_SECRET` | `fireeye-dev-secret-change-me` | Secret for session cookies |

### User Management

Users can be managed in two ways:

#### Option 1: Environment Variable (Recommended for Docker)

Edit `FIREEYE_USERS` in `docker-compose.yml`:

```yaml
environment:
  - FIREEYE_USERS=admin:fireeye,demo:demo123,inspector:inspect2026
```

Then restart: `docker compose up -d`

**For AI agents**: To add a user, append `,newuser:newpass` to the `FIREEYE_USERS` line in `docker-compose.yml` and run `docker compose up -d`.

#### Option 2: JSON File

Create `users.json` in the FireEye directory:

```json
{
  "admin": "fireeye",
  "demo": "demo123",
  "inspector": "inspect2026"
}
```

Uncomment the volume mount in `docker-compose.yml`:
```yaml
- ./users.json:/app/users.json:ro
```

**For AI agents**: Read/write `users.json` with standard JSON operations. The file is hot-reloaded on each login attempt.

### Adding/Removing Users (AI Agent Instructions)

```bash
# Add a user via env var (edit docker-compose.yml)
sed -i 's/FIREEYE_USERS=\(.*\)/FIREEYE_USERS=\1,newuser:newpass/' docker-compose.yml
docker compose up -d

# Or via users.json
python3 -c "
import json
users = json.load(open('users.json'))
users['newuser'] = 'newpass'
json.dump(users, open('users.json', 'w'), indent=2)
"
docker compose restart fireeye

# Remove a user
python3 -c "
import json
users = json.load(open('users.json'))
del users['olduser']
json.dump(users, open('users.json', 'w'), indent=2)
"
docker compose restart fireeye
```

## GPU Support

The Docker image uses NVIDIA CUDA runtime for YOLO GPU inference. Requirements:
- NVIDIA GPU with drivers installed
- `nvidia-container-toolkit` installed and configured
- Docker configured with NVIDIA runtime

```bash
# Install nvidia-container-toolkit (Ubuntu)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  -o /tmp/nvidia-gpg.key
sudo gpg --batch --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  /tmp/nvidia-gpg.key
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Environment Variables

All FireEye settings use the `FIREEYE_` prefix. See `.env.example` for the full list.

Key variables for deployment:
- `FIREEYE_OPENROUTER_API_KEY` — Required for cloud LLM inference
- `FIREEYE_LLM_BACKEND` — `openrouter` (default) or `local`
- `FIREEYE_YOLO_MODEL_NAME` — Path to YOLO model file
- `FIREEYE_YOLO_DEVICE` — `auto`, `cpu`, or `cuda`
