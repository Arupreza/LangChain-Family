use anyhow::{anyhow, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Role {
    System,
    Human,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessageTemplate {
    role: Role,
    text: String, // may contain placeholders like {domain}, {topic}
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatPromptTemplate {
    parts: Vec<ChatMessageTemplate>,
    input_variables: Vec<String>,
}

impl ChatPromptTemplate {
    pub fn new(parts: Vec<(Role, &str)>, input_variables: impl IntoIterator<Item = impl Into<String>>) -> Result<Self> {
        let parts = parts
            .into_iter()
            .map(|(role, text)| ChatMessageTemplate { role, text: text.to_string() })
            .collect::<Vec<_>>();

        let cpt = Self {
            parts,
            input_variables: input_variables.into_iter().map(Into::into).collect(),
        };
        cpt.validate()?;
        Ok(cpt)
    }

    fn validate(&self) -> Result<()> {
        // collect placeholders used in all parts
        let re = Regex::new(r"\{([a-zA-Z0-9_]+)\}")?;
        let mut found: HashSet<String> = HashSet::new();
        for p in &self.parts {
            for cap in re.captures_iter(&p.text) {
                if let Some(m) = cap.get(1) {
                    found.insert(m.as_str().to_string());
                }
            }
        }
        let declared: HashSet<String> = self.input_variables.iter().cloned().collect();

        let missing: Vec<_> = declared.difference(&found).cloned().collect();
        if !missing.is_empty() {
            return Err(anyhow!("Missing variables in template: {:?}", missing));
        }
        let extra: Vec<_> = found.difference(&declared).cloned().collect();
        if !extra.is_empty() {
            return Err(anyhow!("Undeclared variables used in template: {:?}", extra));
        }
        Ok(())
    }

    /// Render to concrete messages with placeholders substituted.
    pub fn invoke<'a>(&self, vars: &HashMap<&'a str, &'a str>) -> Result<Vec<(Role, String)>> {
        for need in &self.input_variables {
            if !vars.contains_key(need.as_str()) {
                return Err(anyhow!("Missing value for variable '{need}'"));
            }
        }
        let mut out = Vec::with_capacity(self.parts.len());
        for p in &self.parts {
            let mut text = p.text.clone();
            for key in &self.input_variables {
                let needle = format!("{{{}}}", key);
                text = text.replace(&needle, vars[key.as_str()]);
            }
            out.push((p.role.clone(), text));
        }
        Ok(out)
    }
}

fn main() -> Result<()> {
    // Equivalent to your Python:
    // ChatPromptTemplate([
    //   ('system', 'You are a helpful {domain} expert'),
    //   ('human',  'Explain in simple terms, what is {topic}')
    // ])
    let chat_template = ChatPromptTemplate::new(
        vec![
            (Role::System, "You are a helpful {domain} expert"),
            (Role::Human,  "Explain in simple terms, what is {topic}"),
        ],
        ["domain", "topic"],
    )?;

    let vars = HashMap::from([("domain", "honey"), ("topic", "history")]);
    let prompt = chat_template.invoke(&vars)?; // Vec<(Role, String)>

    // Print in a readable way (similar to printing the ChatPromptValue)
    println!("--- RENDERED CHAT PROMPT ---");
    for (role, text) in prompt {
        let r = match role { Role::System => "system", Role::Human => "human" };
        println!("{r}: {text}");
    }

    Ok(())
}