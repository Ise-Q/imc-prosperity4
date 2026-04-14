# Setup Instructions

This guide will get you from zero to a working Python environment for the IMC Prosperity 4 competition. Follow every step in order.

---

## Prerequisites

### 1. Git

Make sure you have Git installed. Check by running:

```bash
git --version
```

If not installed:
- **macOS**: `brew install git` or install Xcode Command Line Tools (`xcode-select --install`)
- **Linux (Ubuntu/Debian)**: `sudo apt install git`
- **Windows**: Download from https://git-scm.com/downloads

### 2. Python 3.12+

Check if you have it:

```bash
python3 --version
```

You need **Python 3.12 or newer**. If not:
- **macOS**: `brew install python@3.12`
- **Linux**: `sudo apt install python3.12`
- **Windows**: Download from https://www.python.org/downloads/ (check "Add to PATH" during install)

### 3. uv

`uv` is the tool we use to manage the Python environment and dependencies. It replaces `pip`, `venv`, and `conda` all in one.

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After installation, close and reopen your terminal, then verify:
```bash
uv --version
```

---

## Clone the Repository

```bash
git clone <repo-url>
cd imc-prosperity4
```

Replace `<repo-url>` with the actual GitHub URL shared by the team lead.

---

## Install the Environment

Run this single command inside the project folder:

```bash
uv sync
```

This does three things automatically:
1. Creates a `.venv/` folder in the project directory
2. Installs Python 3.12 if needed
3. Installs all dependencies exactly as specified in `uv.lock`

You should see output like:
```
Resolved N packages in Xs
Installed N packages in Xs
```

**You never need to run `pip install` for this project.** `uv sync` handles everything.

---

## Verify the Setup

Run these two commands to confirm everything is working:

```bash
uv run python --version
```
Expected: `Python 3.12.x`

```bash
uv run python -c "import pandas, numpy, matplotlib; print('All good!')"
```
Expected: `All good!`

---

## Open Jupyter Notebooks

All data analysis is done in Jupyter notebooks. To launch:

```bash
uv run jupyter notebook
```

This will open a browser window at `http://localhost:8888`. Navigate to the round folder and open any `.ipynb` file.

Alternatively, for the newer interface:
```bash
uv run jupyter lab
```

### VS Code Users

If you prefer VS Code with the Jupyter extension:

1. Open the project folder in VS Code
2. Open any `.ipynb` file
3. Click **"Select Kernel"** in the top right
4. Choose **"Python Environments"** → select `.venv` (it should show the path ending in `.venv/bin/python` or `.venv\Scripts\python.exe` on Windows)

The Jupyter extension will use the project's virtual environment automatically.

---

## Adding New Packages

If you need a package that isn't installed (e.g., `scipy`):

```bash
uv add scipy
```

This updates `pyproject.toml` and `uv.lock`. **Commit both files** so your teammates get the same package when they run `uv sync`.

Your teammates then just run:
```bash
git pull
uv sync
```

---

## Staying in Sync with the Team

Whenever you pull the latest changes from the repo:

```bash
git pull
uv sync
```

Always run `uv sync` after pulling — someone may have added or updated a dependency.

---

## Important Rules

| Do this | Not this |
|---|---|
| `uv run python script.py` | `python script.py` |
| `uv run jupyter notebook` | `jupyter notebook` |
| `uv add <package>` | `pip install <package>` |
| `uv sync` after `git pull` | Skip `uv sync` and wonder why imports fail |

**Never manually activate `.venv/`** — the `uv run` prefix handles that for you.

---

## Troubleshooting

**"uv: command not found"** — Close and reopen your terminal after installing uv. On some systems you may need to add `~/.cargo/bin` to your PATH.

**"Python 3.12 not found"** — Run `uv python install 3.12` and then `uv sync` again.

**Jupyter kernel won't start** — Make sure you ran `uv sync` successfully and are launching with `uv run jupyter notebook`.

**Import errors in notebooks** — Confirm the kernel is set to the `.venv` Python (see VS Code instructions above), not your system Python.
