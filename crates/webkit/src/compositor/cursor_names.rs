//! Which named cursor a cursor image is. WPE has no cursor-shape support: it
//! draws each cursor from the `default` Xcursor theme by its CSS name (falling
//! back to Adwaita, as WPE does) and hands the compositor only the image.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::OnceLock;

use xcursor::CursorTheme;

use crate::web_view::CSS_CURSORS;

/// The CSS name whose theme image is `pixels`, an ARGB8888 image `width` wide,
/// rows packed, as Xcursor files store them.
pub(crate) fn identify(width: u32, height: u32, pixels: &[u8]) -> Option<&'static str> {
    images().get(&fingerprint(width, height, pixels)).copied()
}

fn fingerprint(width: u32, height: u32, pixels: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    (width, height, pixels).hash(&mut hasher);
    hasher.finish()
}

/// Every size of every named cursor in the theme, by fingerprint.
fn images() -> &'static HashMap<u64, &'static str> {
    static IMAGES: OnceLock<HashMap<u64, &'static str>> = OnceLock::new();
    IMAGES.get_or_init(|| {
        let themes = [CursorTheme::load("default"), CursorTheme::load("Adwaita")];
        let mut images = HashMap::new();
        for (name, _) in CSS_CURSORS {
            let Some(path) = themes.iter().find_map(|theme| theme.load_icon(name)) else {
                continue;
            };
            let Ok(file) = std::fs::read(path) else {
                continue;
            };
            for image in xcursor::parser::parse_xcursor(&file).unwrap_or_default() {
                images
                    .entry(fingerprint(image.width, image.height, &image.pixels_rgba))
                    .or_insert(*name);
            }
        }
        images
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn identifies_theme_images() {
        let theme = xcursor::CursorTheme::load("default");
        let Some(path) = theme.load_icon("pointer") else {
            return;
        };
        let images = xcursor::parser::parse_xcursor(&std::fs::read(path).unwrap()).unwrap();
        let image = &images[0];
        let name = super::identify(image.width, image.height, &image.pixels_rgba);
        assert!(matches!(name, Some("pointer" | "default")), "{name:?}");
        assert_eq!(super::identify(1, 1, &[0, 0, 0, 0]), None);
    }
}
