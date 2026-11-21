use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use owned_ttf_parser::GlyphId;

use crate::fonts::{self, FaceInfo, LoadedFace, Segment};

/// The families tried for interface text, closest to a Mac's first.
const UI_FAMILIES: &[&str] = &[
    "SF Pro Text",
    "SF Pro",
    "Inter",
    "Cantarell",
    "Noto Sans",
    "Roboto",
    "DejaVu Sans",
    "Liberation Sans",
];

/// Families searched for a character the chosen face lacks: the menu glyphs
/// (command, option, shift) are missing from many interface fonts, and emoji
/// from all of them.
const FALLBACK_FAMILIES: &[&str] = &[
    "DejaVu Sans",
    "Noto Sans Symbols 2",
    "Noto Sans Symbols",
    "Symbola",
    "Apple Color Emoji",
    "Noto Color Emoji",
    "Twemoji",
    "Fluent Color Emoji",
    "Noto Emoji",
];

/// A flag is a pair of these, which an emoji font joins into one glyph.
const REGIONAL_INDICATORS: std::ops::RangeInclusive<char> = '\u{1F1E6}'..='\u{1F1FF}';

/// Outlines by (face, glyph), parsed once.
type Outlines = HashMap<(usize, u16), Arc<Vec<Segment>>>;

/// A face plus the faces searched for characters it lacks.
pub struct Font {
    faces: Vec<Arc<LoadedFace>>,
    outlines: Mutex<Outlines>,
}

/// One glyph ready to draw: its outline in font units, y up.
pub struct Glyph {
    pub outline: Arc<Vec<Segment>>,
    /// Points per font unit.
    pub scale: f32,
    pub advance: f32,
    /// Set for a colour glyph, an emoji's, drawn in its own colours rather
    /// than filled from `outline`.
    pub color: Option<ColorGlyph>,
}

pub struct ColorGlyph {
    pub face: Arc<LoadedFace>,
    pub glyph: GlyphId,
}

/// The glyph two faces' lookups agree on, from the ligatures in `face`'s
/// substitution table: how an emoji font joins a flag's pair.
fn ligature(face: &owned_ttf_parser::Face<'_>, first: GlyphId, second: GlyphId) -> Option<GlyphId> {
    use owned_ttf_parser::gsub::SubstitutionSubtable;
    for lookup in face.tables().gsub?.lookups {
        for subtable in lookup.subtables.into_iter::<SubstitutionSubtable>() {
            let SubstitutionSubtable::Ligature(ligatures) = subtable else {
                continue;
            };
            let Some(set) = ligatures
                .coverage
                .get(first)
                .and_then(|at| ligatures.ligature_sets.get(at))
            else {
                continue;
            };
            if let Some(found) = set
                .into_iter()
                .find(|l| l.components.len() == 1 && l.components.get(0) == Some(second))
            {
                return Some(found.glyph);
            }
        }
    }
    None
}

fn exact_family(family: &str, weight: u16) -> Option<FaceInfo> {
    fonts::all_faces()
        .into_iter()
        .filter(|face| face.family.eq_ignore_ascii_case(family) && !face.italic)
        .min_by_key(|face| face.weight.abs_diff(weight))
}

impl Font {
    /// The interface font at `weight` (400 regular, 700 bold), or `None` on a
    /// host with no usable fonts at all.
    pub fn system(weight: u16) -> Option<Arc<Font>> {
        Font::cached(weight, false)
    }

    /// The desktop's monospace font at `weight`, for commands and paths, or
    /// `None` when the host has none.
    pub fn monospace(weight: u16) -> Option<Arc<Font>> {
        Font::cached(weight, true)
    }

    fn cached(weight: u16, monospace: bool) -> Option<Arc<Font>> {
        type Cache = HashMap<(u16, bool), Option<Arc<Font>>>;
        static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let mut cache = cache.lock().unwrap_or_else(|e| e.into_inner());
        cache
            .entry((weight, monospace))
            .or_insert_with(|| Font::load(weight, monospace).map(Arc::new))
            .clone()
    }

    fn load(weight: u16, monospace: bool) -> Option<Font> {
        let default = || {
            let family = fonts::default_family(monospace)?;
            fonts::resolve(&family, weight, false)
        };
        let primary = match monospace {
            true => default()?,
            false => UI_FAMILIES
                .iter()
                .find_map(|family| exact_family(family, weight))
                .or_else(default)?,
        };
        let mut infos = vec![primary];
        for family in FALLBACK_FAMILIES {
            if let Some(face) = exact_family(family, weight)
                && !infos.iter().any(|known| known.path == face.path)
            {
                infos.push(face);
            }
        }
        let faces: Vec<Arc<LoadedFace>> = infos
            .iter()
            .filter_map(|info| fonts::load(&info.path, info.index))
            .collect();
        Font::from_faces(faces)
    }

    /// [`Font::from_faces`], then the symbol fallbacks the system font has,
    /// for a program's own face that lacks a character the interface needs.
    pub fn from_faces_with_fallback(mut faces: Vec<Arc<LoadedFace>>) -> Option<Font> {
        let fallbacks: Vec<Arc<LoadedFace>> = {
            static FALLBACKS: OnceLock<Vec<Arc<LoadedFace>>> = OnceLock::new();
            FALLBACKS
                .get_or_init(|| {
                    FALLBACK_FAMILIES
                        .iter()
                        .filter_map(|family| exact_family(family, 400))
                        .filter_map(|info| fonts::load(&info.path, info.index))
                        .collect()
                })
                .clone()
        };
        faces.extend(fallbacks);
        Font::from_faces(faces)
    }

    /// A font from faces already loaded, the first preferred.
    pub fn from_faces(faces: Vec<Arc<LoadedFace>>) -> Option<Font> {
        (!faces.is_empty()).then(|| Font {
            faces,
            outlines: Mutex::new(HashMap::new()),
        })
    }

    /// Which face draws `c`, and as which glyph; the primary face's missing
    /// glyph when none has it.
    fn lookup(&self, c: char) -> (usize, GlyphId) {
        for (index, face) in self.faces.iter().enumerate() {
            if let Some(glyph) = face.face().glyph_index(c) {
                return (index, glyph);
            }
        }
        (0, GlyphId(0))
    }

    fn units_per_em(&self, face: usize) -> f32 {
        f32::from(self.faces[face].face().units_per_em().max(1))
    }

    /// The face and glyph each character of `text` is drawn with, a flag's
    /// pair joined into the one glyph its emoji face has for it; the
    /// variation selectors that only ask for emoji or text style draw nothing.
    fn clusters(&self, text: &str) -> Vec<(usize, GlyphId)> {
        let mut out = Vec::with_capacity(text.len());
        let mut chars = text.chars().peekable();
        while let Some(c) = chars.next() {
            if matches!(c, '\u{FE0E}' | '\u{FE0F}') {
                continue;
            }
            if REGIONAL_INDICATORS.contains(&c)
                && let Some(&next) = chars.peek().filter(|n| REGIONAL_INDICATORS.contains(n))
            {
                let flag = self.faces.iter().enumerate().find_map(|(index, face)| {
                    let face = face.face();
                    let joined = ligature(face, face.glyph_index(c)?, face.glyph_index(next)?)?;
                    Some((index, joined))
                });
                if let Some(flag) = flag {
                    chars.next();
                    out.push(flag);
                    continue;
                }
            }
            out.push(self.lookup(c));
        }
        out
    }

    /// The glyphs of `text` at `size` points, in order.
    pub fn glyphs(&self, text: &str, size: f32) -> Vec<Glyph> {
        let mut out = Vec::with_capacity(text.len());
        for (face, glyph) in self.clusters(text) {
            let scale = size / self.units_per_em(face);
            let advance = self.faces[face]
                .face()
                .glyph_hor_advance(glyph)
                .map_or(0.0, f32::from)
                * scale;
            let outline = {
                let mut cache = self.outlines.lock().unwrap_or_else(|e| e.into_inner());
                cache
                    .entry((face, glyph.0))
                    .or_insert_with(|| Arc::new(fonts::outline(self.faces[face].face(), glyph.0)))
                    .clone()
            };
            let color = self.faces[face]
                .face()
                .is_color_glyph(glyph)
                .then(|| ColorGlyph {
                    face: self.faces[face].clone(),
                    glyph,
                });
            out.push(Glyph {
                outline,
                scale,
                advance,
                color,
            });
        }
        out
    }

    /// How far `text` advances at `size` points.
    pub fn measure(&self, text: &str, size: f32) -> f32 {
        self.clusters(text)
            .into_iter()
            .map(|(face, glyph)| {
                let scale = size / self.units_per_em(face);
                self.faces[face]
                    .face()
                    .glyph_hor_advance(glyph)
                    .map_or(0.0, f32::from)
                    * scale
            })
            .sum()
    }

    /// The primary face's ascender and descender at `size`, both positive.
    pub fn metrics(&self, size: f32) -> (f32, f32) {
        let face = self.faces[0].face();
        let scale = size / self.units_per_em(0);
        (
            f32::from(face.ascender()) * scale,
            -f32::from(face.descender()) * scale,
        )
    }

    /// The baseline that centres a line of text vertically in a band
    /// starting at `top` and `height` tall.
    pub fn centered_baseline(&self, size: f32, top: f32, height: f32) -> f32 {
        let (ascent, descent) = self.metrics(size);
        (top + (height + ascent - descent) / 2.0).round()
    }

    /// `text` shortened with an ellipsis to fit `width`, or unchanged when
    /// it already fits.
    pub fn truncate(&self, text: &str, size: f32, width: f32) -> String {
        if self.measure(text, size) <= width {
            return text.to_string();
        }
        let ellipsis = '\u{2026}';
        let mut kept: Vec<char> = text.chars().collect();
        while !kept.is_empty() {
            kept.pop();
            let candidate: String = kept.iter().chain(std::iter::once(&ellipsis)).collect();
            if self.measure(&candidate, size) <= width {
                return candidate;
            }
        }
        String::new()
    }

    /// `text` broken into lines no wider than `width`: at spaces where it
    /// can, and inside a word only when the word alone is too wide, as a
    /// long path is. Each `\n` starts a line, and a blank one is kept.
    pub fn wrap(&self, text: &str, size: f32, width: f32) -> Vec<String> {
        let mut lines = Vec::new();
        for paragraph in text.split('\n') {
            let mut line = String::new();
            for word in paragraph.split_whitespace() {
                let joined = match line.is_empty() {
                    true => word.to_string(),
                    false => format!("{line} {word}"),
                };
                if self.measure(&joined, size) <= width {
                    line = joined;
                    continue;
                }
                if !line.is_empty() {
                    lines.push(std::mem::take(&mut line));
                }
                for c in word.chars() {
                    line.push(c);
                    if line.chars().nth(1).is_some() && self.measure(&line, size) > width {
                        line.pop();
                        lines.push(std::mem::take(&mut line));
                        line.push(c);
                    }
                }
            }
            lines.push(line);
        }
        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncation_keeps_what_fits_and_marks_the_cut() {
        let Some(font) = Font::system(400) else {
            return;
        };
        let text = "A rather long document title";
        assert_eq!(font.truncate(text, 13.0, 10_000.0), text);
        let short = font.truncate(text, 13.0, 80.0);
        assert!(short.ends_with('\u{2026}'), "{short}");
        assert!(font.measure(&short, 13.0) <= 80.0);
        assert_eq!(font.truncate(text, 13.0, 0.0), "");
    }

    #[test]
    fn wrapping_breaks_at_spaces_and_inside_overlong_words() {
        let Some(font) = Font::system(400) else {
            return;
        };
        let width = font.measure("alpha beta", 13.0);
        assert_eq!(
            font.wrap("alpha beta gamma\n\nend", 13.0, width),
            ["alpha beta", "gamma", "", "end"]
        );
        let long = "x".repeat(200);
        let lines = font.wrap(&long, 13.0, width);
        assert!(lines.len() > 1);
        assert_eq!(lines.concat(), long);
        assert!(lines.iter().all(|line| font.measure(line, 13.0) <= width));
    }

    #[test]
    fn text_is_measured_by_its_advances() {
        let Some(font) = Font::system(400) else {
            return;
        };
        let one = font.measure("m", 13.0);
        assert!(one > 0.0);
        assert!((font.measure("mm", 13.0) - 2.0 * one).abs() < 1e-3);
        assert!(font.measure("m", 26.0) > one);
    }

    #[test]
    fn a_flag_is_one_colour_glyph_from_the_emoji_face() {
        let Some(font) = Font::system(400) else {
            return;
        };
        let flag = "\u{1F1E9}\u{1F1EA}";
        let glyphs = font.glyphs(flag, 13.0);
        // Hosts without a colour emoji font draw the two letters instead.
        if glyphs.iter().all(|glyph| glyph.color.is_none()) {
            return;
        }
        assert_eq!(glyphs.len(), 1, "the pair is joined into one flag");
        assert!((font.measure(flag, 13.0) - glyphs[0].advance).abs() < 1e-3);
        // The style selector asks for an emoji and draws nothing itself.
        assert_eq!(font.glyphs("\u{2764}\u{FE0F}", 13.0).len(), 1);
    }
}
