//! Notification banners over the program's own window, for a launcher whose
//! desktop holds its notifications back over a full-screen game.
//!
//! `ROSEMARY_NOTIFICATIONS=PATH`: a socket listened on at `PATH`. Each
//! connection carries one notification: its title, a newline, then its text.
//! The banner is an overlay on a view of its own in the frontmost window, so
//! it stays inside that window, under whatever covers it, and at its top right
//! however it is resized. It slides in from the right, stays as long as a
//! Mac's does and slides out again, or at once when clicked. The timing is
//! measured from a Catalina banner (the Mojave design).

use std::cell::RefCell;
use std::collections::VecDeque;
use std::io::Read;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use chrome::banner::Banner;
use chrome::{Canvas, Image, Rect, View};
use foundation::geometry::{NSPoint, NSRect, NSSize};
use objc::object::Id;

pub const NOTIFICATIONS_ENV: &str = "ROSEMARY_NOTIFICATIONS";

/// Sliding in from the window's edge: quick, then settling.
const ENTER: Duration = Duration::from_millis(400);
const ENTER_CURVE: Curve = Curve(0.4, 0.25, 0.55, 0.95);
/// At rest between the two slides.
const REST: Duration = Duration::from_secs(6);
/// Sliding back out: slow to start, then away.
const LEAVE: Duration = Duration::from_millis(400);
const LEAVE_CURVE: Curve = Curve(0.65, 0.0, 0.9, 0.8);

/// A cubic Bézier timing function through (0, 0) and (1, 1), as Core
/// Animation's: the control points' x and y.
#[derive(Clone, Copy)]
struct Curve(f32, f32, f32, f32);

impl Curve {
    /// How far along the motion is when `time` of it, from 0 to 1, has passed.
    fn at(self, time: f32) -> f32 {
        let Curve(x1, y1, x2, y2) = self;
        let along = |t: f32, a: f32, b: f32| {
            3.0 * (1.0 - t) * (1.0 - t) * t * a + 3.0 * (1.0 - t) * t * t * b + t * t * t
        };
        // x rises with t, so halving finds the t whose x is `time` to well
        // within a pixel in this many steps.
        let (mut low, mut high) = (0.0f32, 1.0f32);
        for _ in 0..24 {
            let middle = (low + high) / 2.0;
            match along(middle, x1, x2) < time.clamp(0.0, 1.0) {
                true => low = middle,
                false => high = middle,
            }
        }
        along((low + high) / 2.0, y1, y2)
    }
}

/// Titles and texts received and not yet shown, oldest first.
static PENDING: Mutex<VecDeque<(String, String)>> = Mutex::new(VecDeque::new());

struct Showing {
    banner: Banner,
    window: Id,
    started: Instant,
    /// When it began sliding out, and how far in it was then.
    leaving: Option<(Instant, f32)>,
    /// The view in the window's content view whose overlay is the banner.
    view: Option<Id>,
    /// The overlay's size in pixels and its theme, as last painted.
    painted: Option<(usize, usize, bool)>,
    /// Where the banner itself is, in the window's points, for clicks.
    card: Option<Rect>,
}

thread_local! {
    static SHOWING: RefCell<Option<Showing>> = const { RefCell::new(None) };
}

/// Listen on `ROSEMARY_NOTIFICATIONS`, when it is set. Called once there is a
/// window to show banners over: the program's windowless helpers, which inherit
/// the variable, must not take the socket from it.
fn listen() {
    let Some(path) = std::env::var_os(NOTIFICATIONS_ENV).filter(|path| !path.is_empty()) else {
        return;
    };
    let path = std::path::PathBuf::from(path);
    if std::os::unix::net::UnixStream::connect(&path).is_ok() {
        eprintln!(
            "rosemary: appkit: {NOTIFICATIONS_ENV}: {} already has a listener; banners go there",
            path.display()
        );
        return;
    }
    // A socket left behind by a session that was killed.
    let _ = std::fs::remove_file(&path);
    let listener = match std::os::unix::net::UnixListener::bind(&path) {
        Ok(listener) => listener,
        Err(error) => {
            eprintln!(
                "rosemary: appkit: {NOTIFICATIONS_ENV}: cannot listen on {}: {error}",
                path.display()
            );
            return;
        }
    };
    let started = std::thread::Builder::new()
        .name("rosemary-notifications".to_string())
        .spawn(move || {
            for mut stream in listener.incoming().flatten() {
                let mut message = String::new();
                if stream.read_to_string(&mut message).is_err() {
                    continue;
                }
                let (title, text) = message.split_once('\n').unwrap_or((&message, ""));
                PENDING
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push_back((title.trim().to_string(), text.trim().to_string()));
            }
        });
    if let Err(error) = started {
        eprintln!("rosemary: appkit: {NOTIFICATIONS_ENV}: no listening thread: {error}");
    }
}

/// Show the next banner, or slide the one showing a step; once per pump.
pub(crate) fn step() {
    SHOWING.with(|showing| {
        let mut showing = showing.borrow_mut();
        if showing.is_none() {
            // Kept until there is a window to show it over.
            let Some(window) = frontmost() else {
                return;
            };
            static LISTENING: std::sync::Once = std::sync::Once::new();
            LISTENING.call_once(listen);
            let Some((title, text)) = PENDING
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .pop_front()
            else {
                return;
            };
            *showing = Some(Showing {
                banner: Banner {
                    title,
                    text,
                    icon: crate::icon::current().map(|icon| Image {
                        width: icon.width as usize,
                        height: icon.height as usize,
                        pixels: icon.pixels,
                    }),
                },
                window,
                started: Instant::now(),
                leaving: None,
                view: None,
                painted: None,
                card: None,
            });
        }
        let done = showing.as_mut().is_some_and(|current| !current.advance());
        if done && let Some(current) = showing.take() {
            if let Some(view) = current.view {
                crate::backing::remove_overlay(view);
                send(view, "removeFromSuperview");
                objc::arc::release(view);
            }
        }
    });
}

/// Whether a button press or release at `x`, `y` in SDL window `id`'s points
/// lands on the banner; then it is the banner's, and a click sends it away as a
/// click on a Mac's does.
pub(crate) fn pointer(id: u32, x: f32, y: f32, clicked: bool) -> bool {
    SHOWING.with(|showing| {
        let mut showing = showing.borrow_mut();
        let Some(current) = showing.as_mut() else {
            return false;
        };
        let on_window = crate::window::facts(current.window)
            .is_some_and(|facts| sdl3::video::window_id(facts.sdl as sdl3::Window) == id);
        if !on_window
            || !current
                .card
                .is_some_and(|card| card.contains(chrome::Point::new(x, y)))
        {
            return false;
        }
        if clicked && current.leaving.is_none() {
            let now = Instant::now();
            let shown = shown(current.started, None, now).unwrap_or(0.0);
            current.leaving = Some((now, shown));
        }
        true
    })
}

/// The key window, else the first titled one shown.
fn frontmost() -> Option<Id> {
    let shown: Vec<(Id, bool)> = crate::window::all()
        .into_iter()
        .filter_map(|window| {
            let facts = crate::window::facts(window)?;
            (facts.shown && facts.titled && !facts.panel).then_some((window, facts.key))
        })
        .collect();
    shown
        .iter()
        .find(|(_, key)| *key)
        .or(shown.first())
        .map(|(window, _)| *window)
}

/// How far in a banner is at `now`, from 0 (beyond the window's edge) to 1
/// (at rest), or `None` once it has slid out: it started at `started`, and
/// began leaving at `leaving` from where it was then, if it was sent away.
fn shown(started: Instant, leaving: Option<(Instant, f32)>, now: Instant) -> Option<f32> {
    let fraction = |from: Instant, length: Duration| {
        now.saturating_duration_since(from).as_secs_f32() / length.as_secs_f32()
    };
    let (left, from) = leaving.unwrap_or((started + ENTER + REST, 1.0));
    if now >= left {
        let gone = fraction(left, LEAVE);
        return (gone < 1.0).then(|| from * (1.0 - LEAVE_CURVE.at(gone)));
    }
    Some(ENTER_CURVE.at(fraction(started, ENTER).min(1.0)))
}

/// `receiver` gets `selector` with no arguments.
fn send(receiver: Id, selector: &str) {
    // SAFETY: a selector that takes no arguments and returns nothing.
    unsafe {
        let call: unsafe extern "C" fn(Id, objc::sel::Sel) =
            std::mem::transmute(objc::msg::entry());
        call(receiver, objc::sel::get(selector));
    }
}

/// A view whose `hitTest:` finds nothing, so the clicks over it reach the
/// window's own views; the banner's own clicks are taken before the program's.
fn banner_view_class() -> &'static objc::class::Class {
    static CLASS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let class = *CLASS.get_or_init(|| {
        let class = corefoundation::base::register(
            c"RosemaryNotificationBanner",
            objc::class::lookup(c"NSView"),
            crate::view::INSTANCE_SIZE,
            false,
        );
        crate::define::methods!(class, {
            "hitTest:" => hit_nothing, c"@32@0:8{CGPoint=dd}16";
        });
        class as *const objc::class::Class as usize
    });
    // SAFETY: registered above, and classes live as long as the process.
    unsafe { &*(class as *const objc::class::Class) }
}

unsafe extern "C" fn hit_nothing(_receiver: Id, _sel: objc::sel::Sel, _point: NSPoint) -> Id {
    std::ptr::null_mut()
}

impl Showing {
    /// Slide and paint one step; false once the banner is gone.
    fn advance(&mut self) -> bool {
        let Some(facts) = crate::window::facts(self.window) else {
            return false;
        };
        let Some(eased) = shown(self.started, self.leaving, Instant::now()) else {
            return false;
        };
        if !facts.shown {
            return false;
        }
        let content = crate::window::content_view_of(self.window);
        if content.is_null() {
            return true;
        }
        let bounds = crate::view::bounds_of(content);
        let (content_width, content_height) = (bounds.size.width as f32, bounds.size.height as f32);
        let theme = crate::decor::theme();
        let layout = self.banner.layout(Rect::default(), theme);
        let (room, origin) = Banner::canvas(&layout, theme);
        // At rest at the content's top right; on the way it starts and ends
        // beyond the right edge, which cuts it.
        let resting = content_width - theme.banner_inset - layout.bounds.width - origin.x;
        let x = resting + (1.0 - eased) * (content_width - resting);
        let top = theme.banner_inset - origin.y;
        let y = match crate::view::is_flipped(content) {
            true => top,
            false => content_height - top - room.height,
        };
        let frame = NSRect {
            origin: NSPoint {
                x: bounds.origin.x + f64::from(x),
                y: bounds.origin.y + f64::from(y),
            },
            size: NSSize {
                width: f64::from(room.width),
                height: f64::from(room.height),
            },
        };
        let view = match self.view {
            Some(view) => {
                if crate::view::frame_of(view) != frame {
                    // SAFETY: a view of this runtime and a frame by value.
                    unsafe {
                        let call: unsafe extern "C" fn(Id, objc::sel::Sel, NSRect) =
                            std::mem::transmute(objc::msg::entry());
                        call(view, objc::sel::get("setFrame:"), frame);
                    }
                }
                view
            }
            None => {
                let view = objc::object::create_instance(banner_view_class(), 0);
                // SAFETY: a new view, a frame by value, and the window's
                // content view to put it in, last so it is over the rest.
                unsafe {
                    let init: unsafe extern "C" fn(Id, objc::sel::Sel, NSRect) -> Id =
                        std::mem::transmute(objc::msg::entry());
                    init(view, objc::sel::get("initWithFrame:"), frame);
                    let add: unsafe extern "C" fn(Id, objc::sel::Sel, Id) =
                        std::mem::transmute(objc::msg::entry());
                    add(content, objc::sel::get("addSubview:"), view);
                }
                self.view = Some(view);
                view
            }
        };
        // The content starts under the titlebar, when the window has one.
        let titlebar = crate::decor::inset_for(
            crate::decor::config(),
            facts.titled,
            facts.panel,
            facts.fullscreen,
        ) as f32;
        self.card = Some(Rect::new(
            x + origin.x,
            titlebar + theme.banner_inset,
            layout.bounds.width,
            layout.bounds.height,
        ));

        let scale = sdl3::video::window_pixel_density(facts.sdl as sdl3::Window).max(1.0);
        let (width, height) = (
            (room.width * scale).ceil() as usize,
            (room.height * scale).ceil() as usize,
        );
        let painted = (width, height, theme.dark);
        if self.painted != Some(painted) {
            self.painted = Some(painted);
            let mut pixels = vec![0u8; width * height * 4];
            let placed = self
                .banner
                .layout(Rect::new(origin.x, origin.y, 0.0, 0.0), theme);
            self.banner.paint(
                &mut Canvas::new(&mut pixels, width, height, width * 4, scale),
                &placed,
                theme,
            );
            crate::backing::set_overlay(
                view,
                crate::backing::Overlay {
                    width: width as u32,
                    height: height as u32,
                    picture: crate::backing::Picture::Pixels(pixels),
                },
            );
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Option<f32>, b: f32) -> bool {
        a.is_some_and(|a| (a - b).abs() < 0.01)
    }

    #[test]
    fn a_banner_slides_in_and_out_as_the_measured_one_did() {
        let start = Instant::now();
        let at = |d: Duration| shown(start, None, start + d);
        assert!(close(at(Duration::ZERO), 0.0));
        // The Catalina banner, every 80 ms of its 400 ms slide in.
        assert!(close(at(ENTER * 3 / 10), 0.30));
        assert!(close(at(ENTER / 2), 0.60));
        assert!(close(at(ENTER * 7 / 10), 0.85));
        assert!(close(at(ENTER), 1.0));
        assert!(close(at(ENTER + REST / 2), 1.0));
        // And out again: 38% of the way 270 ms into its 400 ms.
        assert!(close(at(ENTER + REST + LEAVE * 27 / 40), 1.0 - 0.38));
        assert_eq!(at(ENTER + REST + LEAVE), None);
    }

    #[test]
    fn clicks_over_the_banner_view_reach_the_views_under_it() {
        crate::bootstrap();
        let view = objc::object::create_instance(banner_view_class(), 0);
        // SAFETY: a new banner view, a frame by value, and a point inside it.
        let hit = unsafe {
            let init: unsafe extern "C" fn(Id, objc::sel::Sel, NSRect) -> Id =
                std::mem::transmute(objc::msg::entry());
            let size = NSSize {
                width: 100.0,
                height: 50.0,
            };
            init(
                view,
                objc::sel::get("initWithFrame:"),
                NSRect {
                    origin: NSPoint { x: 0.0, y: 0.0 },
                    size,
                },
            );
            let hit_test: unsafe extern "C" fn(Id, objc::sel::Sel, NSPoint) -> Id =
                std::mem::transmute(objc::msg::entry());
            hit_test(
                view,
                objc::sel::get("hitTest:"),
                NSPoint { x: 10.0, y: 10.0 },
            )
        };
        assert!(hit.is_null(), "the banner view takes no clicks of its own");
        objc::arc::release(view);
    }

    #[test]
    fn a_clicked_banner_slides_out_from_where_it_was() {
        let start = Instant::now();
        let clicked = start + ENTER / 2;
        let from = shown(start, None, clicked).expect("on its way in");
        let leaving = Some((clicked, from));
        assert!(close(shown(start, leaving, clicked), from));
        let halfway = shown(start, leaving, clicked + LEAVE / 2).expect("on its way out");
        assert!(halfway < from && halfway > 0.0);
        assert_eq!(shown(start, leaving, clicked + LEAVE), None);
    }
}
