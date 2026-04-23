# symbolic-priors-cd

# 1. Remove the old virtual environment

Remove-Item -Recurse -Force .venv

# 2. Create a fresh Python 3.12 virtual environment

py -3.12 -m venv .venv

# 3. Upgrade pip inside the virtual environment

.\.venv\Scripts\python.exe -m pip install --upgrade pip

# 4. Install the project in editable mode, including dev dependencies

.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

# 5. Verify core packages import correctly from the virtual environment

.\.venv\Scripts\python.exe -c "import dagma, torch, networkx, pandas, numpy, scipy, igraph, yaml, pydantic, tqdm, wandb, seaborn; print('all imports ok from venv')"

# 6. Confirm pytest is installed in the virtual environment

.\.venv\Scripts\python.exe -m pytest --version

# 7. Confirm the Python interpreter being used is the one inside .venv

.\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"

# 8. Freeze the full working environment to a lock file

.\.venv\Scripts\python.exe -m pip freeze > requirements-lock.txt

# 9. Reinstall after changing pyproject.toml or package structure

.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

# 10. Verify the package import after switching to src/symbolic_priors_cd layout

.\.venv\Scripts\python.exe -c "import symbolic_priors_cd; print('package import ok')"
