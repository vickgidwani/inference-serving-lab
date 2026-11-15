#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${LAB_DIR}/.env.gpu-vm"
DOCKER_LOG="/tmp/inference-lab-dockerd.log"

log() {
  printf '[inference-lab] %s\n' "$*"
}

fail() {
  printf '[inference-lab] ERROR: %s\n' "$*" >&2
  exit 1
}

if [[ "${EUID}" -ne 0 ]]; then
  fail "run this script as root on the GPU VM (for example: sudo bash scripts/setup_gpu_vm.sh)"
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
  fail "nvidia-smi is unavailable; use an Ubuntu GPU VM with an NVIDIA driver installed"
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

if [[ -n "${GPU_VM_TRITON_IMAGE:-}" ]]; then
  triton_image="${GPU_VM_TRITON_IMAGE}"
  log "Using GPU_VM_TRITON_IMAGE override: ${triton_image}"
else
  if (( driver_major >= 615 )); then
    triton_image="nvcr.io/nvidia/tritonserver:26.09-py3"
  elif (( driver_major >= 580 )); then
    triton_image="nvcr.io/nvidia/tritonserver:25.11-py3"
  elif (( driver_major >= 570 )); then
    triton_image="nvcr.io/nvidia/tritonserver:25.03-py3"
  else
    fail "driver ${driver_version} is older than the supported GPU VM profiles (R570+); use a newer NVIDIA driver"
  fi
fi

log "GPU: ${gpu_name}, compute capability ${compute_cap}, driver ${driver_version}"
log "Triton image: ${triton_image}"

export DEBIAN_FRONTEND=noninteractive
packages=()
package_installed() {
  dpkg-query -W -f='${Status}' "$1" 2>/dev/null | grep -q 'install ok installed'
}
python_bin=""
if command -v python >/dev/null 2>&1 && python -m pip --version >/dev/null 2>&1; then
  python_bin="$(command -v python)"
else
  package_installed python3-venv || packages+=(python3-venv)
fi
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

  log "Starting dockerd directly"
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
  fail "dockerd did not become ready; this script requires an Ubuntu GPU VM that allows Docker containers"
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

log "Checking whether this runtime can launch a basic container"
container_check_log="/tmp/inference-lab-container-check.log"
if ! docker run --rm --network host hello-world >"${container_check_log}" 2>&1; then
  cat "${container_check_log}" >&2
  if grep -Eqi 'default sandbox|failed to mount|operation not permitted|invalid argument' "${container_check_log}"; then
    fail "Docker started, but the VM kernel denied container setup; use a standard GPU VM with Docker container privileges"
  fi
  fail "Docker could not launch the hello-world preflight container; inspect the error above"
fi

printf 'TRITON_IMAGE=%s\n' "${triton_image}" > "${ENV_FILE}"
chmod 0644 "${ENV_FILE}"

log "Installing the lab's Python export and client extras"
if [[ -z "${python_bin}" ]]; then
  command -v python3 >/dev/null 2>&1 || fail "Python 3 is not available"
  python3 -m venv --system-site-packages "${LAB_DIR}/.venv"
  python_bin="${LAB_DIR}/.venv/bin/python"
fi
"${python_bin}" -m pip install -e "${LAB_DIR}[export,client]"

if [[ "${python_bin}" == "${LAB_DIR}/.venv/bin/python" ]]; then
  for executable in inference-lab-export inference-lab-bench-ort inference-lab-bench-triton; do
    ln -sfn "${LAB_DIR}/.venv/bin/${executable}" "/usr/local/bin/${executable}"
  done
fi

if [[ -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]] && id -u "${SUDO_USER}" >/dev/null 2>&1; then
  usermod -aG docker "${SUDO_USER}"
  log "Added ${SUDO_USER} to the Docker group; reconnect to use Docker without sudo"
fi

log "Checking GPU access from the selected Triton image (the first run pulls the image)"
if ! docker run --rm --gpus all --entrypoint /bin/bash "${triton_image}" \
  -lc 'test -x /opt/tritonserver/bin/tritonserver && test -c /dev/nvidia0 && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader'; then
  fail "the Triton image could not access the NVIDIA GPU; inspect Docker/NVIDIA runtime setup"
fi

log "Setup complete. Compose will use ${ENV_FILE}."
log "Next: inference-lab-export --model-repo models"
log "Then: docker compose --env-file .env.gpu-vm up"
