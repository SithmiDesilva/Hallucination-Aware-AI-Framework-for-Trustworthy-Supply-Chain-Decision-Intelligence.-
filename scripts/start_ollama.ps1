# Start Ollama with the settings that keep Llama-3.1-8B Q4_K_M fully on the 6 GB RTX 3050 (4096 ctx).
# Quit the Ollama tray app first, then run this in its own window and leave it open.
# Verified 2026-10-30: `ollama ps` -> 100% GPU, 5.0 GB, context 4096.
$env:OLLAMA_FLASH_ATTENTION = "1"
$env:OLLAMA_KV_CACHE_TYPE   = "q8_0"     # halves the context cache (512 -> 272 MiB)
$env:LLAMA_ARG_FIT_TARGET   = "256"      # free-VRAM safety margin in MiB (default 1024 forces CPU offload)
ollama serve
