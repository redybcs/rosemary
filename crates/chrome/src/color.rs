//! Colour glyphs (COLR), as emoji fonts draw them: layers of outlines filled
//! with solid colours and gradients, painted with tiny-skia.

use owned_ttf_parser::colr::{self, CompositeMode, GradientExtend, Paint};
use owned_ttf_parser::{Face, GlyphId, OutlineBuilder, RgbaColor};
use tiny_skia::{
    BlendMode, Color, FillRule, GradientStop, LinearGradient, Mask, Pixmap, PixmapPaint, Point,
    RadialGradient, Shader, SpreadMode, Transform,
};

use crate::paint::Image;

/// `glyph` of `face` painted into a `width` by `height` image, its font units
/// mapped to the image's pixels, y down, by `transform`; `foreground` is the
/// text's colour, for the layers that use it. `None` for a glyph with no
/// colour layers.
pub(crate) fn paint(
    face: &Face<'_>,
    glyph: GlyphId,
    width: u32,
    height: u32,
    transform: Transform,
    foreground: RgbaColor,
) -> Option<Image> {
    let mut painter = Painter {
        face,
        transforms: vec![transform],
        path: None,
        clips: vec![None],
        layers: vec![(Pixmap::new(width, height)?, CompositeMode::SourceOver)],
        width,
        height,
    };
    face.paint_color_glyph(glyph, 0, foreground, &mut painter)?;
    let (pixmap, _) = painter.layers.into_iter().next()?;
    // tiny-skia keeps premultiplied pixels; an Image is straight.
    let pixels = pixmap
        .pixels()
        .iter()
        .flat_map(|pixel| {
            let color = pixel.demultiply();
            [color.red(), color.green(), color.blue(), color.alpha()]
        })
        .collect();
    Some(Image {
        width: width as usize,
        height: height as usize,
        pixels,
    })
}

struct Painter<'f, 'a> {
    face: &'f Face<'a>,
    /// Font units to pixels, innermost last.
    transforms: Vec<Transform>,
    /// The outline last stored, in pixels.
    path: Option<tiny_skia::Path>,
    /// What painting is confined to, innermost last; `None` is everywhere.
    clips: Vec<Option<Mask>>,
    /// Layers being painted, the image itself first, and how each is put
    /// down on the one under it.
    layers: Vec<(Pixmap, CompositeMode)>,
    width: u32,
    height: u32,
}

impl Painter<'_, '_> {
    fn transform(&self) -> Transform {
        *self
            .transforms
            .last()
            .expect("the outermost transform is never popped")
    }

    fn clip_to(&mut self, path: Option<tiny_skia::Path>) {
        let current = self.clips.last().cloned().flatten();
        let clip = match (path, current) {
            (Some(path), Some(mut mask)) => {
                mask.intersect_path(&path, FillRule::Winding, true, Transform::identity());
                Some(mask)
            }
            (Some(path), None) => Mask::new(self.width, self.height).map(|mut mask| {
                mask.fill_path(&path, FillRule::Winding, true, Transform::identity());
                mask
            }),
            // Nothing to clip to: nothing is painted until it is popped.
            (None, _) => Mask::new(self.width, self.height),
        };
        self.clips.push(clip);
    }

    fn shader(&self, paint: &Paint<'_>) -> Option<Shader<'static>> {
        let coords = self.face.variation_coordinates();
        let stops = |stops: &mut dyn Iterator<Item = colr::ColorStop>| -> Vec<GradientStop> {
            stops
                .map(|stop| GradientStop::new(stop.stop_offset, color(stop.color)))
                .collect()
        };
        let transform = self.transform();
        match paint {
            Paint::Solid(c) => Some(Shader::SolidColor(color(*c))),
            Paint::LinearGradient(g) => {
                // The gradient runs from p0 towards p1, turned to stand square
                // on the line from p0 to p2.
                let (dx, dy) = (g.x2 - g.x0, g.y2 - g.y0);
                let along = ((g.x1 - g.x0) * -dy + (g.y1 - g.y0) * dx)
                    / (dx * dx + dy * dy).max(f32::EPSILON);
                let end = Point::from_xy(g.x0 - dy * along, g.y0 + dx * along);
                LinearGradient::new(
                    Point::from_xy(g.x0, g.y0),
                    end,
                    stops(&mut g.stops(0, coords)),
                    spread(g.extend),
                    transform,
                )
            }
            Paint::RadialGradient(g) => RadialGradient::new(
                Point::from_xy(g.x0, g.y0),
                Point::from_xy(g.x1, g.y1),
                g.r1,
                stops(&mut g.stops(0, coords)),
                spread(g.extend),
                transform,
            ),
            // tiny-skia has no sweep: its first colour stands in.
            Paint::SweepGradient(g) => g
                .stops(0, coords)
                .next()
                .map(|stop| Shader::SolidColor(color(stop.color))),
        }
    }
}

fn color(c: RgbaColor) -> Color {
    Color::from_rgba8(c.red, c.green, c.blue, c.alpha)
}

fn spread(extend: GradientExtend) -> SpreadMode {
    match extend {
        GradientExtend::Pad => SpreadMode::Pad,
        GradientExtend::Repeat => SpreadMode::Repeat,
        GradientExtend::Reflect => SpreadMode::Reflect,
    }
}

fn blend(mode: CompositeMode) -> BlendMode {
    match mode {
        CompositeMode::Clear => BlendMode::Clear,
        CompositeMode::Source => BlendMode::Source,
        CompositeMode::Destination => BlendMode::Destination,
        CompositeMode::SourceOver => BlendMode::SourceOver,
        CompositeMode::DestinationOver => BlendMode::DestinationOver,
        CompositeMode::SourceIn => BlendMode::SourceIn,
        CompositeMode::DestinationIn => BlendMode::DestinationIn,
        CompositeMode::SourceOut => BlendMode::SourceOut,
        CompositeMode::DestinationOut => BlendMode::DestinationOut,
        CompositeMode::SourceAtop => BlendMode::SourceAtop,
        CompositeMode::DestinationAtop => BlendMode::DestinationAtop,
        CompositeMode::Xor => BlendMode::Xor,
        CompositeMode::Plus => BlendMode::Plus,
        CompositeMode::Screen => BlendMode::Screen,
        CompositeMode::Overlay => BlendMode::Overlay,
        CompositeMode::Darken => BlendMode::Darken,
        CompositeMode::Lighten => BlendMode::Lighten,
        CompositeMode::ColorDodge => BlendMode::ColorDodge,
        CompositeMode::ColorBurn => BlendMode::ColorBurn,
        CompositeMode::HardLight => BlendMode::HardLight,
        CompositeMode::SoftLight => BlendMode::SoftLight,
        CompositeMode::Difference => BlendMode::Difference,
        CompositeMode::Exclusion => BlendMode::Exclusion,
        CompositeMode::Multiply => BlendMode::Multiply,
        CompositeMode::Hue => BlendMode::Hue,
        CompositeMode::Saturation => BlendMode::Saturation,
        CompositeMode::Color => BlendMode::Color,
        CompositeMode::Luminosity => BlendMode::Luminosity,
    }
}

/// An outline in font units, as tiny-skia builds it.
struct Outline(tiny_skia::PathBuilder);

impl OutlineBuilder for Outline {
    fn move_to(&mut self, x: f32, y: f32) {
        self.0.move_to(x, y);
    }
    fn line_to(&mut self, x: f32, y: f32) {
        self.0.line_to(x, y);
    }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        self.0.quad_to(x1, y1, x, y);
    }
    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        self.0.cubic_to(x1, y1, x2, y2, x, y);
    }
    fn close(&mut self) {
        self.0.close();
    }
}

impl<'a> colr::Painter<'a> for Painter<'_, 'a> {
    fn outline_glyph(&mut self, glyph: GlyphId) {
        let mut outline = Outline(tiny_skia::PathBuilder::new());
        self.path = self
            .face
            .outline_glyph(glyph, &mut outline)
            .and_then(|_| outline.0.finish())
            .and_then(|path| path.transform(self.transform()));
    }

    fn paint(&mut self, paint: Paint<'a>) {
        let (Some(path), Some(shader)) = (self.path.clone(), self.shader(&paint)) else {
            return;
        };
        let mask = self.clips.last().cloned().flatten();
        let mut brush = tiny_skia::Paint::default();
        brush.shader = shader;
        brush.anti_alias = true;
        let (layer, _) = self
            .layers
            .last_mut()
            .expect("the image itself is never popped");
        layer.fill_path(
            &path,
            &brush,
            FillRule::Winding,
            Transform::identity(),
            mask.as_ref(),
        );
    }

    fn push_clip(&mut self) {
        self.clip_to(self.path.clone());
    }

    fn push_clip_box(&mut self, clipbox: colr::ClipBox) {
        let rect =
            tiny_skia::Rect::from_ltrb(clipbox.x_min, clipbox.y_min, clipbox.x_max, clipbox.y_max)
                .map(tiny_skia::PathBuilder::from_rect)
                .and_then(|path| path.transform(self.transform()));
        self.clip_to(rect);
    }

    fn pop_clip(&mut self) {
        if self.clips.len() > 1 {
            self.clips.pop();
        }
    }

    fn push_layer(&mut self, mode: CompositeMode) {
        if let Some(layer) = Pixmap::new(self.width, self.height) {
            self.layers.push((layer, mode));
        }
    }

    fn pop_layer(&mut self) {
        if self.layers.len() < 2 {
            return;
        }
        let (layer, mode) = self.layers.pop().expect("checked above");
        let (under, _) = self.layers.last_mut().expect("checked above");
        let paint = PixmapPaint {
            blend_mode: blend(mode),
            ..PixmapPaint::default()
        };
        under.draw_pixmap(0, 0, layer.as_ref(), &paint, Transform::identity(), None);
    }

    fn push_transform(&mut self, t: owned_ttf_parser::Transform) {
        let inner = Transform::from_row(t.a, t.b, t.c, t.d, t.e, t.f);
        self.transforms.push(self.transform().pre_concat(inner));
    }

    fn pop_transform(&mut self) {
        if self.transforms.len() > 1 {
            self.transforms.pop();
        }
    }
}
