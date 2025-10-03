.PHONY: build
build:
	cargo build --target wasm32-unknown-unknown --release
	wasm-bindgen \
		target/wasm32-unknown-unknown/release/mathmap_wasm.wasm \
		--target web \
		--out-dir web/pkg

.PHONY: serve
serve:
	python3 -m http.server --directory web --bind 127.0.0.1
