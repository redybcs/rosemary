//! Wall time to translate each MSL source given, averaged over repeated runs.
//!
//! `cargo run --release -p metal_translate --example translate_cost -- shader.metal...`

use std::time::Instant;

fn main() {
    let target = metal_translate::TargetProfile::vulkan_1_3();
    let rounds: u32 = std::env::var("ROUNDS")
        .ok()
        .and_then(|r| r.parse().ok())
        .unwrap_or(50);
    for path in std::env::args().skip(1) {
        let source = std::fs::read_to_string(&path).expect("a readable source");
        let translated = match metal_translate::translate_msl(&source, &target) {
            Ok(translated) => translated,
            Err(error) => {
                eprintln!("{path}: {error}");
                continue;
            }
        };
        let words: usize = translated.entries.iter().map(|e| e.words.len()).sum();
        if let Some(dump) = std::env::var_os("DUMP") {
            let mut dump = std::path::PathBuf::from(dump);
            dump.push(
                std::path::Path::new(&path)
                    .file_name()
                    .expect("a file name"),
            );
            let bytes: Vec<u8> = translated
                .entries
                .iter()
                .flat_map(|e| e.words.iter().flat_map(|w| w.to_le_bytes()))
                .collect();
            std::fs::write(dump.with_extension("spv"), bytes).expect("a writable dump");
        }
        let start = Instant::now();
        for _ in 0..rounds {
            std::hint::black_box(metal_translate::translate_msl(&source, &target).ok());
        }
        println!(
            "{path}: {} entries, {words} words, {:.1} us per translation",
            translated.entries.len(),
            start.elapsed().as_secs_f64() * 1e6 / f64::from(rounds)
        );
    }
}
