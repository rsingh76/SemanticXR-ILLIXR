# Building & Running `semantic_python` against the `sceneGraphDemo` conda env

This document records how the `semantic_python` plugin (branch `semantic_python_plugin`)
was built and test-run against the conda environment **`sceneGraphDemo`** on this machine.

- **Branch:** `semantic_python_plugin`
- **Conda env:** `$CONDA_PREFIX` (Python **3.10.19**)
- **Build dir:** `build-scene/`
- **Plugin docs:** [docs/docs/plugin_README/README_semantic_python.md](docs/docs/plugin_README/README_semantic_python.md)

---

## 0. What the plugin is

`semantic_python` embeds a CPython interpreter (via pybind11) in a thread and runs a user
Python script. It injects three proxy objects into the script's globals (no import needed):

| Proxy | Purpose |
|---|---|
| `illixr_semantic_reader` | reads `semantic_data` frames (RGB+depth+poses) |
| `illixr_voice_reader` | reads `voice_query` objects |
| `illixr_response_writer` | writes `query_response` results back **over the network** |

Two env vars drive the script:

- `SEMANTIC_PYTHON_SCRIPT` — absolute path to the script to run (**required**)
- `SEMANTIC_PYTHON_ARGS` — optional comma-separated `key=value`/flag args

Because `illixr_response_writer` is a **network** writer, the plugin requires a network
backend (`tcp_network_backend`) to be loaded alongside it.

---

## 1. Environment verification

The conda env was confirmed to have everything needed:

```
Python                3.10.19
libpython             $CONDA_PREFIX/lib/libpython3.10.so.1.0   (full .so.1.0, not a symlink)
pybind11              3.0.4   (cmake dir: .../site-packages/pybind11/share/cmake/pybind11)
numpy                 1.24.3
```

> **Note:** The example script `plugins/semantic_python/semanticxr.py` originally hardcoded
> `libpython3.12.so.1.0` in its `ctypes` preload snippet. This was changed to
> **`libpython3.10.so.1.0`** to match `sceneGraphDemo`. If you switch to a different
> Python version, update that line accordingly.

---

## 2. Build

Configured with the `semantic_python` profile and pybind11 pointed explicitly at the conda
env. The duplicated `PYTHON_*` / `Python_*` flags cover CMake's capitalization differences;
the "PYTHON_EXECUTABLE / PYTHON_ROOT_DIR were not used" warning is harmless.

```bash
cd /path/to/ILLIXR
ENV=$CONDA_PREFIX

cmake -B build-scene -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DYAML_FILE="$PWD/profiles/semantic_python.yaml" \
  -DDOWNLOAD_DATA_FILE=OFF \
  -DPYTHON_EXECUTABLE="$ENV/bin/python"  -DPython_EXECUTABLE="$ENV/bin/python" \
  -DPYTHON_ROOT_DIR="$ENV"               -DPython_ROOT_DIR="$ENV" \
  -Dpybind11_DIR="$ENV/lib/python3.10/site-packages/pybind11/share/cmake/pybind11"

cmake --build build-scene -j"$(nproc)"
```

This enables only `semantic_python` + `tcp_network_backend` (plus the ILLIXR core).
`-DDOWNLOAD_DATA_FILE=OFF` skips the EuRoC dataset download (not needed for this plugin).

### Build verification

The configure step reported:

```
-- Found Python: .../sceneGraphDemo/bin/python (found suitable version "3.10.19" ...)
                 found components: Interpreter Development.Module Development.Embed
-- Found pybind11: .../sceneGraphDemo/lib/python3.10/site-packages/pybind11/include (version "3.0.4")
```

Artifacts produced under `build-scene/`:

```
main.opt.exe
plugins/semantic_python/libplugin.semantic_python.opt.so
plugins/tcp_network_backend/libplugin.tcp_network_backend.opt.so
```

The plugin links the **conda** libpython (confirmed with `readelf -d`):

```
readelf -d build-scene/plugins/semantic_python/libplugin.semantic_python.opt.so | grep NEEDED
  ... NEEDED  Shared library: [libpython3.10.so.1.0]
```

---

## 3. Running (from the build dir, no system install)

The runtime `dlopen`s `libplugin.<name>.opt.so` by name, searching `LD_LIBRARY_PATH`
(see [src/plugin.cpp:302-319](src/plugin.cpp#L302-L319)). Since we did **not** run
`cmake --install`, we add the build's plugin dirs and the conda lib dir to `LD_LIBRARY_PATH`.

```bash
cd /path/to/ILLIXR/build-scene
ENV=$CONDA_PREFIX

# --- Python / conda env ---
export PYTHONHOME="$ENV"
export PYTHONPATH="$ENV/lib/python3.10/site-packages"
export VIRTUAL_ENV="$ENV"          # lets the plugin's venv-style site-packages helper fire
export LD_LIBRARY_PATH="$ENV/lib:$PWD/plugins/semantic_python:$PWD/plugins/tcp_network_backend:$LD_LIBRARY_PATH"
# REQUIRED — see "Troubleshooting" below. Makes libpython symbols global so Python
# C-extensions (_ctypes, numpy, ...) can import inside the embedded interpreter.
export LD_PRELOAD="$ENV/lib/libpython3.10.so.1.0:$LD_PRELOAD"

# --- The Python script to run ---
export SEMANTIC_PYTHON_SCRIPT="$PWD/../plugins/semantic_python/semanticxr.py"
export SEMANTIC_PYTHON_ARGS="log_dir=/tmp/illixr_logs"

# --- tcp_network_backend (server) config ---
# The server BINDS to ILLIXR_TCP_SERVER_IP:ILLIXR_TCP_SERVER_PORT, so this must be
# THIS machine's reachable IP (the client is on a different machine).
export ILLIXR_TCP_SERVER_IP=YOUR_HOST_IP ILLIXR_TCP_SERVER_PORT=50057
# CLIENT_IP/PORT are NOT used in server mode (only the client side uses them as its
# local bind address); left here only for completeness.
export ILLIXR_TCP_CLIENT_IP=YOUR_HOST_IP ILLIXR_TCP_CLIENT_PORT=50058
export ILLIXR_IS_CLIENT=0          # 0 = run as server
export ILLIXR_DISPLAY_MODE=none

./main.opt.exe --plugins=tcp_network_backend,semantic_python --duration=1000
```

> **Server vs client IP:** In server mode (`ILLIXR_IS_CLIENT=0`) only `ILLIXR_TCP_SERVER_IP`
> matters — the server binds to it. Because the client is on a **different machine**, this
> must be this host's externally reachable IP: **`YOUR_HOST_IP`** (not `127.0.0.1`).
> The client machine must in turn set its `ILLIXR_TCP_SERVER_IP=YOUR_HOST_IP` and the same
> `ILLIXR_TCP_SERVER_PORT` to connect.
>
> **Ports:** on Android-based headsets, use ports above 49152 (the README used 50057/50058).
> Make sure any host firewall allows inbound TCP on `ILLIXR_TCP_SERVER_PORT`.

---

## 4. Important behavior observed: the server blocks waiting for a peer

When run as the **server** (`ILLIXR_IS_CLIENT=0`), startup reaches:

```
[tcp_network_backend] Using TCP server IP YOUR_HOST_IP / port 50057 ...
[tcp_network_backend] Is client
```

…and then **blocks**. `tcp_network_backend::start_server()` calls `socket_accept()`
(see [plugins/tcp_network_backend/plugin.cpp:73-81](plugins/tcp_network_backend/plugin.cpp#L73-L81)),
which waits for a client to connect. **Until a peer connects, the Python interpreter does
not launch** — this is expected, because `semantic_python` publishes a networked topic and
the backend won't proceed without its peer.

To actually exercise the Python bridge you therefore need one of:

1. **The real client** (the Unity / headset app) connecting to `SERVER_IP:SERVER_PORT`.
2. **A second ILLIXR instance** run as the client (`ILLIXR_IS_CLIENT=1`) for a loopback test.
3. An interpreter-only test harness (requires a small code change, since the plugin
   constructs a `get_network_writer` that needs the network backend).

---

## 4a. Troubleshooting: `undefined symbol: PyUnicode_FromFormat`

Symptom (after the client connects, when the Python script imports `ctypes`/`numpy`/etc.):

```
[python_bridge] Python exception: ImportError:
  .../lib/python3.10/lib-dynload/_ctypes.cpython-310-x86_64-linux-gnu.so:
  undefined symbol: PyUnicode_FromFormat
```

**Cause:** ILLIXR `dlopen`s plugins with `RTLD_LOCAL` (see
[include/illixr/dynamic_lib.hpp:57](include/illixr/dynamic_lib.hpp#L57)), so the plugin's
`libpython3.10.so.1.0` dependency is loaded into a *local* symbol scope. When the embedded
interpreter then `dlopen`s C-extension modules, they cannot find libpython's symbols
(`PyUnicode_FromFormat`, etc.) in the global scope.

The `ctypes.CDLL(..., RTLD_GLOBAL)` snippet at the top of `semanticxr.py` was meant to fix
this, but it **cannot** — the failure occurs on `import ctypes` itself, before that snippet
can run. (The snippet is now effectively redundant.)

**Fix:** preload libpython so its symbols are global from process start:

```bash
export LD_PRELOAD="$ENV/lib/libpython3.10.so.1.0:$LD_PRELOAD"
```

This is already included in the run recipe above and in `run_semantic_server.sh`.

---

## 5. Quick reference: rebuild & relaunch

```bash
# rebuild after code changes
cmake --build build-scene -j"$(nproc)"

# relaunch server (env vars from section 3 must be set)
./main.opt.exe --plugins=tcp_network_backend,semantic_python --duration=1000
```
