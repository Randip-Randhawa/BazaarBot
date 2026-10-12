# BazaarBot

## Setup

Requires Python 3.12.

Create the virtual environment:

```bash
python -m venv myenv
```

Activate it:

```powershell
# Windows (PowerShell)
myenv\Scripts\Activate.ps1
```

```bash
# Windows (Git Bash)
source myenv/Scripts/activate

# macOS / Linux
source myenv/bin/activate
```

If PowerShell blocks the activation script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

Install dependencies and run the tests:

```bash
python -m pip install -r requirements.txt
python -m pytest
```
