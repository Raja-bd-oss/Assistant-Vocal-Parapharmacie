#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::process::{Command, Stdio};
use std::io::Write;

#[tauri::command]
fn run_python() -> String {
    let mut child = Command::new("python")
        .arg("agent AI/scripts/Assistant_ai.py")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("Erreur lancement script Python");

    let input = r#"{"cmd":"process_audio"}"#;

    child.stdin.as_mut().unwrap().write_all(input.as_bytes()).unwrap();
    child.stdin.as_mut().unwrap().write_all(b"\n").unwrap();

    let output = child.wait_with_output().expect("Erreur lecture sortie");
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![run_python])
        .run(tauri::generate_context!())
        .expect("Erreur démarrage Tauri");
}
