Local LLaMA/TinyLlama with Rust and langchain-rust
This project demonstrates how to run a local LLaMA or TinyLlama model using Rust with the langchain-rust library. Two setups are supported:

A) Ollama + GGUF: Recommended for models in .gguf format or if you can convert to it.
B) vLLM + safetensors: Use for Hugging Face checkpoints (e.g., model.safetensors, tokenizer.json).

The Rust code is identical for both setups; only the api_base and model_name differ.
Prerequisites

Rust: Install via rustup.
Verify: rustc --version and cargo --version.


Local LLM runtime:
Ollama: For .gguf models.
vLLM: For Hugging Face safetensors models.



Setup Instructions
A) Ollama (GGUF)

Install and Start Ollama:

On Linux/macOS, follow instructions on the Ollama website or use a package manager.
Start the server: ollama serve


Get a GGUF Model:

Option 1: Download a prebuilt GGUF (e.g., TinyLlama from TheBloke).
Option 2: Convert a Hugging Face model to GGUF using llama.cpp’s converter.


Create a Modelfile:
cat > Modelfile << 'EOF'
FROM /absolute/path/to/your_model.Q4_K_M.gguf

# Optional: Chat template for TinyLlama/Zephyr-style chat
TEMPLATE """<|system|>
{{ .System }}</s>
<|user|>
{{ .Prompt }}</s>
<|assistant|>"""

PARAMETER temperature 0.5
PARAMETER num_ctx 2048
EOF


Register the Model:
ollama create my-llama-local -f Modelfile
ollama list  # Should show my-llama-local
ollama run my-llama-local


Default Endpoint:

http://127.0.0.1:11434/v1
Change host/port if needed: OLLAMA_HOST=0.0.0.0 OLLAMA_PORT=12345 ollama serve
Update Rust code to use http://127.0.0.1:12345/v1.



B) vLLM (safetensors)

Install vLLM:
python -m pip install "vllm>=0.5.0"
# For CPU-only (slower): python -m pip install "vllm[cpu]>=0.5.0"


Serve Your Model:
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/your_hf_model_dir \
  --tokenizer /path/to/your_hf_model_dir \
  --served-model-name my-ft-llama \
  --host 0.0.0.0 --port 8000 --dtype auto --max-model-len 4096


Default Endpoint:

http://127.0.0.1:8000/v1
Verify: curl http://127.0.0.1:8000/v1/models



C) Rust Project Setup (Common to Both)

Create Cargo.toml:
[package]
name = "local_llama_rust"
version = "0.1.0"
edition = "2021"

[dependencies]
langchain-rust = "4"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
serde_json = "1"
anyhow = "1"


Create src/main.rs:
use langchain_rust::language_models::llm::LLM;
use langchain_rust::llm::openai::{OpenAI, OpenAIConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Choose ONE of the two blocks:
    // A) Ollama (GGUF)
    // let api_base = "http://127.0.0.1:11434/v1";
    // let model_name = "my-llama-local";

    // B) vLLM (safetensors)
    // let api_base = "http://127.0.0.1:8000/v1";
    // let model_name = "my-ft-llama";

    // Example configuration (update as needed):
    let api_base = "http://127.0.0.1:11434/v1";
    let model_name = "my-llama-local".to_string();

    let llm = OpenAI::default()
        .with_config(
            OpenAIConfig::default()
                .with_api_base(api_base)
                .with_api_key("local"), // Not used by Ollama/vLLM
        )
        .with_model(model_name);

    let system = "You are a precise assistant. Answer concisely.";
    let user = "What is the capital of Bangladesh?";
    let prompt = format!(
        "<|system|>\n{}\n</s>\n<|user|>\n{}\n</s>\n<|assistant|>",
        system, user
    );

    let out = llm.invoke(&prompt).await?;
    println!("{}", out);

    Ok(())
}


Build and Run:
cargo run --release



Request Parameters

Ollama: Set defaults in the Modelfile (e.g., PARAMETER temperature 0.5).
vLLM: Supports OpenAI-style parameters (temperature, max_tokens) per request. For fine-grained control:
Set server defaults, or
Use reqwest to call /v1/chat/completions with a custom JSON body.



Verify Model

Ollama:ollama list
ollama show my-llama-local  # Check for "From /path/to/model.gguf"


vLLM:
Verify paths in the startup command (--model, --tokenizer).
Check: curl http://127.0.0.1:8000/v1/models



Troubleshooting

Model not found:
Ollama: Ensure ollama list shows your model; recreate with ollama create.
vLLM: Confirm --served-model-name matches .with_model(); verify server is running.


Cargo errors: Use cargo run --release (double dash).
Anyhow missing: Add anyhow = "1" to Cargo.toml.
No output/wrong port: Confirm api_base matches the server port (11434 for Ollama, 8000 for vLLM unless changed).

Changing Host/Port

Ollama:OLLAMA_HOST=0.0.0.0 OLLAMA_PORT=12345 ollama serve

Update Rust: .with_api_base("http://127.0.0.1:12345/v1").
vLLM:Start with --host 0.0.0.0 --port 9000, then use .with_api_base("http://127.0.0.1:9000/v1").