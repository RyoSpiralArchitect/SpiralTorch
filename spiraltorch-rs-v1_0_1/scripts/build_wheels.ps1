pip install -U pip maturin
Push-Location bindings/st-py
maturin build --release -m pyproject.toml
Pop-Location
