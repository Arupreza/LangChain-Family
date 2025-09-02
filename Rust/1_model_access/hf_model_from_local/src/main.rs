// src/main.rs
use langchain_rust::language_models::llm::LLM;
use langchain_rust::llm::openai::{OpenAI, OpenAIConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let llm = OpenAI::default()
        .with_config(
            OpenAIConfig::default()
                .with_api_base("http://127.0.0.1:11434/v1")
                .with_api_key("ollama"), // not used by Ollama
        )
        .with_model("tinyllama-local".to_string()); // name from `ollama create`

    let prompt = "What is the capital of Bangladesh?";
    let out = llm.invoke(prompt).await?;
    println!("{}", out);
    Ok(())
}
