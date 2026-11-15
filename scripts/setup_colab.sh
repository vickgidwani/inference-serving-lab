#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${LAB_DIR}/.env.colab"
DOCKER_LOG="/tmp/inference-lab-dockerd.log"

log() {
  printf '[inference-lab] %s\n' "$*"
}

fail() {
  printf '[inference-lab] ERROR: %s\n' "$*" >&2
  exit 1
}

if [[ "${EUID}" -ne 0 ]]; then
  fail "run this script as root in the Colab runtime (for example: sudo bash scripts/setup_colab.sh)"
fi

if [[ ! -r /etc/os-release ]]; then
  fail "cannot identify the runtime OS"
fi
# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "24.04" ]]; then
  fail "expected Ubuntu 24.04; found ${PRETTY_NAME:-unknown}. Review the package steps before adapting this script."
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  fail "nvidia-smi is unavailable; connect a GPU runtime before setup"
fi

gpu_row="$(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -n 1)"
IFS=',' read -r gpu_name compute_cap driver_version <<< "${gpu_row}"
gpu_name="$(xargs <<< "${gpu_name}")"
compute_cap="$(xargs <<< "${compute_cap}")"
driver_version="$(xargs <<< "${driver_version}")"
compute_cap_number="${compute_cap//./}"
driver_major="${driver_version%%.*}"

[[ "${compute_cap_number}" =~ ^[0-9]+$ ]] || fail "could not parse GPU compute capability: ${compute_cap}"
[[ "${driver_major}" =~ ^[0-9]+$ ]] || fail "could not parse NVIDIA driver version: ${driver_version}"
(( compute_cap_number >= 75 )) || fail "Triton images in this lab require compute capability 7.5 or newer; found ${compute_cap}"

if [[ -n "${COLAB_TRITON_IMAGE:-}" ]]; then
  triton_image="${COLAB_TRITON_IMAGE}"
  log "Using COLAB_TRITON_IMAGE override: ${triton_image}"
else
  if (( driver_major >= 615 )); then
    triton_image="nvcr.io/nvidia/tritonserver:26.09-py3"
  elif (( driver_major >= 580 )); then
    triton_image="nvcr.io/nvidia/tritonserver:25.11-py3"
  elif (( driver_major >= 570 )); then
    triton_image="nvcr.io/nvidia/tritonserver:25.03-py3"
  else
    fail "driver ${driver_version} is older than the supported Colab profiles (R570+); use a newer runtime or a GCP GPU VM"
  fi
fi

log "GPU: ${gpu_name}, compute capability ${compute_cap}, driver ${driver_version}"
log "Triton image: ${triton_image}"

export DEBIAN_FRONTEND=noninteractive
packages=()
package_installed() {
  dpkg-query -W -f='${Status}' "$1" 2>/dev/null | grep -q 'install ok installed'
}
for package in ca-certificates curl gnupg; do
  package_installed "${package}" || packages+=("${package}")
done
if ! command -v docker >/dev/null 2>&1 || ! command -v dockerd >/dev/null 2>&1; then
  package_installed docker.io || packages+=(docker.io)
fi
if ! docker compose version >/dev/null 2>&1; then
  package_installed docker-compose-v2 || packages+=(docker-compose-v2)
fi
if ((${#packages[@]} > 0)); then
  log "Installing Docker and Compose packages from Ubuntu"
  apt-get update
  apt-get install -y --no-install-recommends "${packages[@]}"
fi

if ! command -v nvidia-ctk >/dev/null 2>&1; then
  log "Installing NVIDIA Container Toolkit from NVIDIA's production apt repository"
  keyring="/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
  repo_file="/etc/apt/sources.list.d/nvidia-container-toolkit.list"
  install -m 0755 -d /usr/share/keyrings
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor --yes -o "${keyring}"
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed "s#deb https://#deb [signed-by=${keyring}] https://#g" \
    > "${repo_file}"
  apt-get update
  apt-get install -y --no-install-recommends nvidia-container-toolkit
fi

if ! command -v nvidia-ctk >/dev/null 2>&1; then
  fail "nvidia-container-toolkit installation did not provide nvidia-ctk"
fi

log "Configuring Docker's NVIDIA runtime"
nvidia-ctk runtime configure --runtime=docker

docker_info_ready() {
  docker info >/dev/null 2>&1
}

docker_gpu_runtime_ready() {
  docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'
}

start_docker() {
  if docker_info_ready; then
    return
  fi

  if command -v systemctl >/dev/null 2>&1; then
    systemctl start docker >/dev/null 2>&1 || true
  fi
  if command -v service >/dev/null 2>&1; then
    service docker start >/dev/null 2>&1 || true
  fi
  if docker_info_ready; then
    return
  fi

  log "Starting dockerd directly (Colab often has no systemd service manager)"
  mkdir -p /var/run/docker
  nohup dockerd --host=unix:///var/run/docker.sock --pidfile=/var/run/docker.pid \
    >"${DOCKER_LOG}" 2>&1 </dev/null &

  for _ in $(seq 1 45); do
    if docker_info_ready; then
      return
    fi
    sleep 1
  done

  tail -n 60 "${DOCKER_LOG}" >&2 || true
  fail "dockerd did not become ready; Colab may restrict nested Docker. A GCP GPU VM is the reliable fallback."
}

start_docker

if ! docker_gpu_runtime_ready; then
  running_containers="$(docker ps -q)"
  if [[ -n "${running_containers}" ]]; then
    fail "Docker is running without the NVIDIA runtime while containers are active; stop them before rerunning setup"
  fi

  log "Restarting an existing idle Docker daemon to load the NVIDIA runtime"
  if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet docker; then
    systemctl restart docker >/dev/null 2>&1 || true
  elif command -v service >/dev/null 2>&1; then
    service docker restart >/dev/null 2>&1 || true
  fi
  if ! docker_gpu_runtime_ready; then
    if command -v pkill >/dev/null 2>&1; then
      pkill -TERM -x dockerd >/dev/null 2>&1 || true
    fi
    for _ in $(seq 1 10); do
      docker_info_ready || break
      sleep 1
    done
    start_docker
  fi
fi

docker compose version >/dev/null 2>&1 || fail "Docker Compose v2 is not available"
docker_gpu_runtime_ready || fail "Docker did not load the NVIDIA runtime after configuration"

printf 'TRITON_IMAGE=%s\n' "${triton_image}" > "${ENV_FILE}"
chmod 0644 "${ENV_FILE}"

log "Installing the lab's Python export and client extras"
python -m pip install -e "${LAB_DIR}[export,client]"

log "Checking GPU access from the selected Triton image (the first run pulls the image)"
if ! docker run --rm --gpus all --entrypoint /bin/bash "${triton_image}" \
  -lc 'test -e /dev/nvidia0 && /opt/tritonserver/bin/tritonserver --version'; then
  fail "the Triton image could not access the T4; inspect Docker/NVIDIA runtime setup or use a GCP GPU VM"
fi

log "Setup complete. Compose will use ${ENV_FILE}."
log "Next: inference-lab-export --model-repo models"
log "Then: docker compose --env-file .env.colab up"
