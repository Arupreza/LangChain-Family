use dotenv::dotenv;
use std::env;
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct HFRequest {
    inputs: String,
    parameters: HFParameters,
}

#[derive(Serialize)]
struct HFParameters {
    max_new_tokens: u32,
    temperature: f32,
}

#[derive(Deserialize, Debug)]
struct HFResponse {
    generated_text: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load environment variables from `.env`
    dotenv().ok();

    // Read token from .env
    let hf_token = env::var("HUGGINGFACEHUB_API_TOKEN")
        .expect("HUGGINGFACEHUB_API_TOKEN must be set");

    // Hugging Face Inference API endpoint for your model
    let repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";
    let url = format!("https://api-inference.huggingface.co/models/{}", repo_id);

    // Build the request payload
    let request = HFRequest {
        inputs: "What is the capital of Bangladesh?".to_string(),
        parameters: HFParameters {
            max_new_tokens: 128,
            temperature: 0.2,
        },
    };

    // Create an HTTP client
    let client = Client::new();

    // Send POST request with Bearer token
    let response = client
        .post(&url)
        .bearer_auth(hf_token)
        .json(&request)
        .send()
        .await?;

    // Hugging Face returns a list of objects
    let json: serde_json::Value = response.json().await?;

    // Extract result (generated text)
    if let Some(arr) = json.as_array() {
        if let Some(obj) = arr.get(0) {
            if let Some(output) = obj.get("generated_text") {
                println!("{}", output.as_str().unwrap_or(""));
            }
        }
    }

    Ok(())
}
