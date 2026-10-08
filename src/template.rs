//! Tiny `{{ .Var }}` template renderer (enough for chat templates).
//! Vars: `.System`, `.Prompt`, `.Messages` (joined), plus any extra keys.

use std::collections::HashMap;

pub fn render(tpl: &str, vars: &HashMap<String, String>) -> String {
    let mut out = tpl.to_string();
    for (k, v) in vars {
        for key in [format!("{{{{ .{k} }}}}", k = k), format!("{{{{.{k}}}}}", k = k)] {
            out = out.replace(&key, v);
        }
    }
    out
}

pub fn default_chat_template() -> &'static str {
    "{{ .System }}\n{{ .Prompt }}"
}

pub fn apply(
    template: Option<&str>,
    system: Option<&str>,
    prompt: &str,
    messages: Option<&str>,
) -> String {
    let tpl = template.unwrap_or(default_chat_template());
    let mut vars = HashMap::new();
    vars.insert("System".into(), system.unwrap_or("").to_string());
    vars.insert("Prompt".into(), prompt.to_string());
    vars.insert("Messages".into(), messages.unwrap_or("").to_string());
    render(tpl, &vars)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn renders_vars() {
        let mut v = std::collections::HashMap::new();
        v.insert("System".into(), "S".into());
        v.insert("Prompt".into(), "P".into());
        assert_eq!(render("{{ .System }}|{{.Prompt}}", &v), "S|P");
    }
}
