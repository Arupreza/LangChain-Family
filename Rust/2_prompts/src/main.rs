use llm_chain::prompt::{Data, StringTemplate};
use llm_chain::traits::Template; // <-- correct place for Template trait
use std::collections::HashMap;
use std::fs;

fn main() -> anyhow::Result<()> {
    // Define template string
    let template_str = r#"
Please summarize the research paper titled "{paper_input}" with the following specifications:
Explanation Style: {style_input}  
Explanation Length: {length_input}  
1. Mathematical Details:  
    - Include relevant mathematical equations if present in the paper.  
    - Explain the mathematical concepts using simple, intuitive code snippets where applicable.  
2. Analogies:  
    - Use relatable analogies to simplify complex ideas.  
If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.  
Ensure the summary is clear, accurate, and aligned with the provided style and length.
"#;

    // Build a StringTemplate
    let string_template = StringTemplate::tera(template_str);

    // Wrap in Data
    let template: Data<StringTemplate> = Data::text(string_template);

    // Prepare variables
    let mut vars = HashMap::new();
    vars.insert("paper_input".to_string(), "EfficientNet: Rethinking Model Scaling for CNNs".to_string());
    vars.insert("style_input".to_string(), "Educational".to_string());
    vars.insert("length_input".to_string(), "Medium".to_string());

    // Render the template
    let rendered = match &template {
        Data::Text(st) => st.render(&vars)?, // ✅ now works
        _ => String::new(),
    };

    println!("Rendered Prompt:\n{}", rendered);

    // Save the original template string to a file
    fs::write("template.json", template_str)?;

    Ok(())
}
