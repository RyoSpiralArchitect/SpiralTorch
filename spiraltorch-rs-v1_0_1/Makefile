.PHONY: test build-wheel dev-install clean

test:
	cargo test -p st-core

dev-install:
	cd bindings/st-py && maturin develop -m pyproject.toml

build-wheel:
	cd bindings/st-py && maturin build --release -m pyproject.toml

clean:
	cargo clean
	rm -rf bindings/st-py/target
