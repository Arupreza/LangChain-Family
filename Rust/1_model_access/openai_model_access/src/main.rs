// Import dotenv crate to load environment variables from a `.env` file
use dotenv::dotenv;

// Import standard library functionality for reading environment variables
use std::env;

// Import OpenAI client and configuration struct from langchain-rust
use langchain_rust::llm::openai::{OpenAI, OpenAIConfig};

// Import the LLM trait so we can call `.invoke()` on the OpenAI client
use langchain_rust::language_models::llm::LLM; 

// Entry point of the async Rust program
#[tokio::main]
async fn main() {
    // Load the `.env` file (if present in the project root)
    // This allows us to keep API keys outside of the code
    dotenv().ok();

    // Fetch the OpenAI API key from environment variables
    // If it's missing, the program will panic with the message
    let api_key = env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY must be set");

    // Build an OpenAI client with configuration
    // Only API key and API base URL are configurable here
    let open_ai = OpenAI::default().with_config(
        OpenAIConfig::default()
            .with_api_key(api_key)  // Add your API key
            .with_api_base("https://api.openai.com/v1"), // OpenAI endpoint
    );

    // Call the `.invoke()` method on the client
    // `.invoke()` is provided by the `LLM` trait we imported
    // It sends the given text prompt to the model and waits for the response
    let response = open_ai
        .invoke("What is the capital of Bangladesh?")
        .await
        .unwrap(); // If an error happens, panic

    // Print the model's response to stdout
    println!("{}", response);
}