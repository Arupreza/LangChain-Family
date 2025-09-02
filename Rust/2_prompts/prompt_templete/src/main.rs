use anyhow::{anyhow, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs, path::Path};

#[derive(Debug, Serialize, Deserialize)]
struct PromptTemplate {
    template: String,
    input_variables: Vec<String>,
    validate_template: bool,
}

impl PromptTemplate {
    fn new(template: impl Into<String>, input_variables: Vec<&str>, validate_template: bool) -> Result<Self> {
        let tpl = template.into();
        let iv: Vec<String> = input_variables.into_iter().map(|s| s.to_string()).collect();
        let pt = Self {
            template: tpl,
            input_variables: iv,
            validate_template,
        };
        if pt.validate_template {
            pt.validate()?;
        }
        Ok(pt)
    }

    /// Ensure that all and only declared variables appear in `{var}` form.
    fn validate(&self) -> Result<()> {
        let re = Regex::new(r"\{([a-zA-Z0-9_]+)\}")?;
        let found: std::collections::HashSet<String> = re
            .captures_iter(&self.template)
            .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
            .collect();

        let declared: std::collections::HashSet<String> = self.input_variables.iter().cloned().collect();

        // Check missing
        let missing: Vec<_> = declared.difference(&found).cloned().collect();
        if !missing.is_empty() {
            return Err(anyhow!("Missing variables in template: {:?}", missing));
        }
        // Check extra
        let extra: Vec<_> = found.difference(&declared).cloned().collect();
        if !extra.is_empty() {
            return Err(anyhow!("Undeclared variables used in template: {:?}", extra));
        }
        Ok(())
    }

    /// Render by substituting `{var}` with provided values.
    fn render(&self, vars: &HashMap<&str, &str>) -> Result<String> {
        // quick sanity: ensure all required vars provided
        for need in &self.input_variables {
            if !vars.contains_key(need.as_str()) {
                return Err(anyhow!("Missing value for variable '{need}'"));
            }
        }
        // Replace each {var}. We only substitute declared vars for safety.
        let mut out = self.template.clone();
        for key in &self.input_variables {
            // Use literal replacement (no regex in the replacement phase)
            let needle = format!("{{{key}}}");
            let val = vars.get(key.as_str()).copied().unwrap_or_default();
            out = out.replace(&needle, val);
        }
        Ok(out)
    }

    /// Save the template metadata as JSON (like `template.save('template.json')`).
    fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }
}

fn main() -> Result<()> {
    // ---- Define the template (mirrors your Python version) ----
    let template_text = r#"
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

    let prompt = PromptTemplate::new(
        template_text,
        vec!["paper_input", "style_input", "length_input"],
        true, // validate_template
    )?;

    // ---- Save to JSON (equivalent to Python's template.save('template.json')) ----
    prompt.save("template.json")?;

    // ---- Example render (optional) ----
    let mut vars = HashMap::new();
    vars.insert("paper_input", "Attention Is All You Need (Vaswani et al., 2017)");
    vars.insert("style_input", "Concise and technical");
    vars.insert("length_input", "About 200 words");

    let rendered = prompt.render(&vars)?;
    println!("--- RENDERED PROMPT ---\n{}", rendered);

    // From here, pass `rendered` into your LLM client (e.g., llm-chain or your own API client).
    Ok(())
}