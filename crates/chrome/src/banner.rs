//! A notification banner, after Mojave's: the app's icon on the left, the
//! title in semibold and the text under it, on the titlebar's gradient with a
//! soft shadow round it.

use crate::alert::{Line, cap};
use crate::geometry::{Point, Rect, Size};
use crate::paint::{Canvas, Image, Path};
use crate::text::Font;
use crate::theme::Theme;
use crate::view::View;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Banner {
    pub title: String,
    pub text: String,
    pub icon: Option<Image>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BannerLayout {
    /// The banner itself; its shadow spreads `banner_shadow_radius` beyond.
    pub bounds: Rect,
    pub icon: Option<Rect>,
    pub lines: Vec<Line>,
}

/// A banner shows one line of title and two of text, cut with an ellipsis.
const TITLE_LINES: usize = 1;
const TEXT_LINES: usize = 2;
/// Rings the shadow is built from; more only smooths what the eye cannot see.
const SHADOW_RINGS: usize = 8;
/// The title is semibold; the nearest weight the host has stands in.
const TITLE_WEIGHT: u16 = 600;
const TEXT_WEIGHT: u16 = 400;

impl Banner {
    /// The room, in points, a banner laid out at `layout` needs with its
    /// shadow, and where in that room the banner goes.
    pub fn canvas(layout: &BannerLayout, theme: &Theme) -> (Size, Point) {
        // The shadow spreads its radius and is dropped by its offset besides.
        let margin = theme.banner_shadow_radius + theme.banner_shadow_offset;
        (
            Size::new(
                layout.bounds.width + 2.0 * margin,
                layout.bounds.height + 2.0 * margin,
            ),
            Point::new(margin, margin),
        )
    }
}

impl View for Banner {
    /// The whole banner: a click dismisses it.
    type Part = ();
    type Layout = BannerLayout;

    /// Only `bounds`' origin is used: a banner is as tall as what it says.
    fn layout(&self, bounds: Rect, theme: &Theme) -> BannerLayout {
        let (x, y) = (bounds.x, bounds.y);
        let width = theme.banner_width;
        let pad = theme.banner_padding;
        let pad_y = theme.banner_padding_vertical;
        let size = theme.banner_icon_size;
        let icon = self
            .icon
            .as_ref()
            .map(|_| Rect::new(x + pad, y + pad_y, size, size));
        let left = icon.map_or(x + pad, |icon| icon.right() + pad);
        let text_width = x + width - pad - left;
        let height = theme.banner_line_height;

        let mut lines = Vec::new();
        let mut text_height = 0.0;
        for (body, bold, font_size, limit) in [
            (&self.title, true, theme.banner_title_size, TITLE_LINES),
            (&self.text, false, theme.banner_text_size, TEXT_LINES),
        ] {
            let Some(font) = Font::system(if bold { TITLE_WEIGHT } else { TEXT_WEIGHT }) else {
                continue;
            };
            for line in cap(font.wrap(body, font_size, text_width), limit) {
                lines.push(Line {
                    origin: Point::new(
                        left,
                        font.centered_baseline(font_size, text_height, height),
                    ),
                    text: line.trim().to_string(),
                    bold,
                });
                text_height += height;
            }
        }
        // Text shorter than the icon sits level with its middle.
        let content = icon.map_or(text_height, |icon| icon.height.max(text_height));
        let top = y + pad_y + (content - text_height) / 2.0;
        for line in &mut lines {
            line.origin.y += top;
        }
        BannerLayout {
            bounds: Rect::new(x, y, width, (content + 2.0 * pad_y).ceil()),
            icon: icon.map(|icon| icon.offset(0.0, (content - icon.height) / 2.0)),
            lines,
        }
    }

    fn paint(&self, canvas: &mut Canvas, layout: &BannerLayout, theme: &Theme) {
        let bounds = layout.bounds;
        let radius = theme.banner_corner_radius;
        // Rings, each a little wider and fainter, that add up darkest at the
        // edge; dropped a little, as light from above casts it.
        let spread = theme.banner_shadow_radius;
        let ring = theme
            .banner_shadow
            .with_alpha(theme.banner_shadow.a / SHADOW_RINGS as f32);
        // From the banner's own edge out to just inside the spread, so the
        // faintest ring still fits in the room left for it.
        for step in (0..SHADOW_RINGS).rev() {
            let grow = spread * step as f32 / SHADOW_RINGS as f32;
            let rect = bounds
                .inset(-grow, -grow)
                .offset(0.0, theme.banner_shadow_offset);
            canvas.fill_rounded_rect(rect, radius + grow, ring);
        }
        let card = Path::rounded_rect(bounds, radius, false);
        canvas.fill_path_vertical_gradient(
            &card,
            bounds,
            theme.titlebar_top,
            theme.titlebar_bottom,
        );
        canvas.stroke_rounded_rect(bounds, radius, 1.0 / canvas.scale(), theme.banner_border);

        if let (Some(image), Some(rect)) = (&self.icon, layout.icon) {
            canvas.draw_image(image, rect);
        }
        for line in &layout.lines {
            let (weight, size) = match line.bold {
                true => (TITLE_WEIGHT, theme.banner_title_size),
                false => (TEXT_WEIGHT, theme.banner_text_size),
            };
            if let Some(font) = Font::system(weight) {
                canvas.draw_text(&line.text, &font, size, line.origin, theme.banner_text);
            }
        }
    }

    fn hit(&self, layout: &BannerLayout, point: Point) -> Option<()> {
        layout.bounds.contains(point).then_some(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn banner(text: &str) -> Banner {
        Banner {
            title: "Server region".to_string(),
            text: text.to_string(),
            icon: Some(Image {
                width: 2,
                height: 2,
                pixels: vec![255; 16],
            }),
        }
    }

    #[test]
    fn a_title_and_a_line_are_a_catalina_banners_height_and_long_text_is_cut_at_two_lines() {
        let theme = Theme::light();
        if Font::system(TEXT_WEIGHT).is_none() {
            return;
        }
        let short = banner("Frankfurt").layout(Rect::default(), &theme);
        assert_eq!(short.bounds.width, theme.banner_width);
        // A title and one line of text, as the measured banner had.
        let text = 2.0 * theme.banner_line_height;
        assert_eq!(
            short.bounds.height,
            (theme.banner_icon_size.max(text) + 2.0 * theme.banner_padding_vertical).ceil()
        );
        let long =
            banner(&"Frankfurt am Main, Germany ".repeat(20)).layout(Rect::default(), &theme);
        let text_lines = long.lines.iter().filter(|l| !l.bold).count();
        assert_eq!(text_lines, TEXT_LINES);
        assert!(
            long.lines
                .last()
                .is_some_and(|l| l.text.ends_with('\u{2026}'))
        );
        assert!(long.bounds.height >= short.bounds.height);
    }

    #[test]
    fn the_banner_is_the_titlebar_gradient_inside_a_shadow_that_fades_out() {
        for theme in [Theme::light(), Theme::dark()] {
            let banner = banner("Frankfurt");
            let layout = banner.layout(Rect::default(), &theme);
            let (size, origin) = Banner::canvas(&layout, &theme);
            let layout = banner.layout(Rect::new(origin.x, origin.y, 0.0, 0.0), &theme);
            let (width, height) = (size.width.ceil() as usize, size.height.ceil() as usize);
            let mut pixels = vec![0u8; width * height * 4];
            banner.paint(
                &mut Canvas::new(&mut pixels, width, height, width * 4, 1.0),
                &layout,
                &theme,
            );
            let at = |x: f32, y: f32| {
                let i = (y as usize * width + x as usize) * 4;
                [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
            };
            // Inside, clear of the rounded corner and below the text: the
            // gradient, opaque.
            let corner = theme.banner_corner_radius;
            let inside = at(
                layout.bounds.right() - corner - 2.0,
                layout.bounds.bottom() - 3.0,
            );
            assert_eq!(inside[3], 255, "the banner is opaque");
            let expected = (theme.titlebar_bottom.b * 255.0).round() as i32;
            assert!(
                (i32::from(inside[0]) - expected).abs() <= 3,
                "{inside:?} is the gradient's foot"
            );
            // The corner of the room is beyond the shadow.
            assert_eq!(at(0.0, 0.0)[3], 0, "the shadow fades out before the edge");
            // Painted with room to spare around the room, nothing lands
            // outside it: the shadow, dropped as it is, is never cut off.
            let spare = 4;
            let (wide, tall) = (width + 2 * spare, height + 2 * spare);
            let mut roomy = vec![0u8; wide * tall * 4];
            let shifted = banner.layout(
                Rect::new(origin.x + spare as f32, origin.y + spare as f32, 0.0, 0.0),
                &theme,
            );
            banner.paint(
                &mut Canvas::new(&mut roomy, wide, tall, wide * 4, 1.0),
                &shifted,
                &theme,
            );
            let outside = (0..tall)
                .flat_map(|y| (0..wide).map(move |x| (x, y)))
                .filter(|&(x, y)| {
                    x < spare || y < spare || x >= spare + width || y >= spare + height
                });
            assert!(
                outside
                    .into_iter()
                    .all(|(x, y)| roomy[(y * wide + x) * 4 + 3] == 0),
                "the shadow fits in the room Banner::canvas leaves"
            );
            // Just outside the banner, the shadow shows.
            let shadow = at(layout.bounds.center().x, layout.bounds.bottom() + 1.0);
            assert!(shadow[3] > 0, "a shadow under the banner");
        }
    }
}
