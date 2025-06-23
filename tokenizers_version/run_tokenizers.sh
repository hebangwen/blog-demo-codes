versions="0.19.1 0.20.0 0.21.1"
for version in $versions; do
    pip install tokenizers==$version 2>&1 > /dev/null
    echo ====== $version ======
    python load_qwen3_tokenizer.py
    echo ====================
done
