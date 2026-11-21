//! The presentation upscalers: their compute shaders, compiled to SPIR-V with
//! glslc, and RAVU-Zoom's two trained weight tables, converted from the mpv
//! hook they ship in to the half floats the shader samples.
//!
//! Without glslc the crate still builds, with no upscaler: presentation then
//! scales a smaller drawable with the plain linear blit it always had.

use std::path::{Path, PathBuf};
use std::process::Command;

const SHADERS: [(&str, &str, &[&str]); 3] = [
    ("sgsr.comp", "sgsr.spv", &["-DSTUD_SGSR_EDGE_DIRECTION"]),
    ("ravu_ar.comp", "ravu_ar.spv", &[]),
    ("sharpen.comp", "sharpen.spv", &[]),
];
const HOOK: &str = "ravu-zoom-ar-r2-rgb.hook";

fn main() {
    let source = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("upscale");
    let out = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    println!("cargo:rerun-if-env-changed=GLSLC");
    for (file, _, _) in SHADERS {
        println!("cargo:rerun-if-changed={}", source.join(file).display());
    }
    println!("cargo:rerun-if-changed={}", source.join(HOOK).display());

    let available = compile_shaders(&source, &out) && write_tables(&source.join(HOOK), &out);
    if !available {
        std::fs::write(
            out.join("ravu_size.rs"),
            "pub const RAVU_LUT_SIZE: (u32, u32) = (0, 0);\n",
        )
        .unwrap();
    }
    let mut module = format!("pub const AVAILABLE: bool = {available};\n");
    for (_, spv, _) in SHADERS {
        let name = spv.trim_end_matches(".spv").to_uppercase();
        module += &match available {
            true => format!(
                "pub static {name}: &[u8] = include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{spv}\"));\n"
            ),
            false => format!("pub static {name}: &[u8] = &[];\n"),
        };
    }
    for (name, file) in [
        ("RAVU_LUT", "ravu_lut.bin"),
        ("RAVU_LUT_AR", "ravu_lut_ar.bin"),
    ] {
        module += &match available {
            true => format!(
                "pub static {name}: &[u8] = include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{file}\"));\n"
            ),
            false => format!("pub static {name}: &[u8] = &[];\n"),
        };
    }
    std::fs::write(out.join("upscale.rs"), module).unwrap();
}

fn compile_shaders(source: &Path, out: &Path) -> bool {
    let glslc = std::env::var_os("GLSLC").unwrap_or_else(|| "glslc".into());
    for (file, spv, defines) in SHADERS {
        let status = Command::new(&glslc)
            .args(["-fshader-stage=compute", "--target-env=vulkan1.1", "-O"])
            .args(defines)
            .arg(source.join(file))
            .arg("-o")
            .arg(out.join(spv))
            .status();
        match status {
            Ok(status) if status.success() => {}
            Ok(status) => panic!("glslc failed on {file}: {status}"),
            Err(error) => {
                println!("cargo:warning=no glslc ({error}): building without the upscalers");
                return false;
            }
        }
    }
    true
}

/// The hook's `//!TEXTURE` blocks, each a declared size and one long hex line
/// of float32 RGBA, as half floats: what mpv uploads them as, and what the
/// weights were trained against. Every size is read from the hook.
fn write_tables(hook: &Path, out: &Path) -> bool {
    let text = std::fs::read_to_string(hook).expect("the RAVU hook");
    let mut tables = Vec::new();
    let mut size = None;
    let mut in_texture = false;
    for line in text.lines() {
        if line.starts_with("//!TEXTURE") {
            in_texture = true;
            size = None;
        } else if let Some(rest) = line.strip_prefix("//!SIZE") {
            let dims: Vec<usize> = rest
                .split_whitespace()
                .map(|v| v.parse().unwrap())
                .collect();
            size = Some((dims[0], dims[1]));
        } else if in_texture && !line.starts_with("//!") && size.is_some() {
            let (width, height) = size.take().unwrap();
            let bytes: Vec<u8> = (0..line.trim().len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&line.trim()[i..i + 2], 16).unwrap())
                .collect();
            assert_eq!(
                bytes.len(),
                width * height * 16,
                "{}: a table is not RGBA float32",
                hook.display()
            );
            let halves: Vec<u8> = bytes
                .chunks_exact(4)
                .flat_map(|v| {
                    half::f16::from_f32(f32::from_le_bytes(v.try_into().unwrap())).to_le_bytes()
                })
                .collect();
            tables.push((width, height, halves));
            in_texture = false;
        }
    }
    assert_eq!(
        tables.len(),
        2,
        "{}: expected the weight and anti-ringing tables",
        hook.display()
    );
    let (width, height, _) = tables[0];
    std::fs::write(out.join("ravu_lut.bin"), &tables[0].2).unwrap();
    std::fs::write(out.join("ravu_lut_ar.bin"), &tables[1].2).unwrap();
    std::fs::write(
        out.join("ravu_size.rs"),
        format!("pub const RAVU_LUT_SIZE: (u32, u32) = ({width}, {height});\n"),
    )
    .unwrap();
    true
}
