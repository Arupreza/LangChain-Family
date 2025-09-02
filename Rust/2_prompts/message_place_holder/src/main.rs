use anyhow::{anyhow, Result};
use std::{fs, path::Path};

/// Split into lines, preserving trailing newlines (like Python readlines()).
fn lines_with_endings(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    for ch in s.chars() {
        buf.push(ch);
        if ch == '\n' {
            out.push(std::mem::take(&mut buf));
        }
    }
    if !buf.is_empty() {
        out.push(buf);
    }
    out
}

/// Escape into a Python-style single-quoted string, showing \n literally.
fn python_repr_string(s: &str) -> String {
    let mut r = String::new();
    r.push('\'');
    for ch in s.chars() {
        match ch {
            '\\' => r.push_str(r"\\"),
            '\'' => r.push_str(r"\'"),
            '\n' => r.push_str(r"\n"),
            _ => r.push(ch),
        }
    }
    r.push('\'');
    r
}

/// Load history lines and normalize: all lines end with `\n` except the last.
fn load_chat_history_file(path: impl AsRef<Path>) -> Result<Vec<String>> {
    let content = fs::read_to_string(&path)
        .map_err(|e| anyhow!("Failed to read {}: {e}", path.as_ref().display()))?;
    let mut lines = lines_with_endings(&content);

    for i in 0..lines.len() {
        let is_last = i + 1 == lines.len();
        if !is_last && !lines[i].ends_with('\n') {
            lines[i].push('\n');
        }
    }
    Ok(lines)
}

fn main() -> Result<()> {
    // Path to your history file
    let history_path = "chat_history.txt";

    // Query
    let query = "Where is my refund?";

    // 1) Load history
    let history_lines = load_chat_history_file(history_path)?;

    // 2) Print Chat history like Python repr
    let history_repr_items: Vec<String> =
        history_lines.iter().map(|s| python_repr_string(s)).collect();
    println!("Chat history: [{}]", history_repr_items.join(", "));

    // 3) Build messages=[...]
    let mut message_parts = Vec::new();

    // SystemMessage
    let system_content = "You are a helpful customer support agent";
    message_parts.push(format!(
        "SystemMessage(content={}, additional_kwargs={{}}, response_metadata={{}})",
        python_repr_string(system_content)
    ));

    // History as HumanMessages
    for line in &history_lines {
        message_parts.push(format!(
            "HumanMessage(content={}, additional_kwargs={{}}, response_metadata={{}})",
            python_repr_string(line)
        ));
    }

    // Final query as HumanMessage
    message_parts.push(format!(
        "HumanMessage(content={}, additional_kwargs={{}}, response_metadata={{}})",
        python_repr_string(query)
    ));

    println!("messages=[{}]", message_parts.join(", "));

    Ok(())
}