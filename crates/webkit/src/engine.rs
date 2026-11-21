use std::cell::RefCell;
use std::ffi::c_void;
use std::sync::OnceLock;

use objc::object::Id;

use crate::compositor::{Composition, Compositor};
use crate::wpe::{self, Api, Ptr};

pub struct Engine {
    pub api: &'static Api,
    /// Where WPE's views are shown: every view's pictures and input go
    /// through it.
    pub compositor: Compositor,
    /// The `WPEDisplay` on the compositor that every view is made on.
    pub display: usize,
    /// The `WebKitNetworkSession` every view shares, as every `WKWebView` with
    /// the default data store shares one cookie jar.
    pub session: usize,
    thread: u32,
}

static ENGINE: OnceLock<Result<Engine, String>> = OnceLock::new();

/// The engine, started on first use; `None` where WPE WebKit is missing or
/// cannot open a display, or on a thread other than the one it runs on.
pub fn get() -> Option<&'static Engine> {
    let engine = ENGINE.get_or_init(|| {
        let started = start();
        if let Err(why) = &started {
            eprintln!("rosemary: webkit: web views will have no content: {why}");
        }
        started
    });
    let engine = engine.as_ref().ok()?;
    if engine.thread != darwin::pthread::id() {
        warn_once("a web view was used off the thread WPE WebKit runs on; ignored");
        return None;
    }
    Some(engine)
}

/// The engine if something has already started it, without starting it.
fn running() -> Option<&'static Engine> {
    let engine = ENGINE.get()?.as_ref().ok()?;
    (engine.thread == darwin::pthread::id()).then_some(engine)
}

fn start() -> Result<Engine, String> {
    if std::env::var_os("ROSEMARY_WEBKIT").is_some_and(|value| value == "0") {
        return Err("disabled by ROSEMARY_WEBKIT=0".to_string());
    }
    let api = wpe::api()?;
    // WebKit names its sandboxed processes after the program; Flatpak's portal only accepts the app id.
    if let Some(app_id) = std::env::var_os("FLATPAK_ID") {
        let app_id = wpe::c_string(&app_id.to_string_lossy());
        // SAFETY: GLib copies the name.
        unsafe { (api.g_set_prgname)(app_id.as_ptr()) };
    }
    let composition = match std::env::var("ROSEMARY_WEBKIT_GPU").as_deref() {
        Ok("0") => Composition::Cpu,
        _ => Composition::Gpu(appkit::backing::importer()),
    };
    let compositor = Compositor::start(wake, composition)?;
    // WPE's display is the compositor, never the host's.
    let socket = wpe::c_string(&compositor.socket.to_string_lossy());
    // SAFETY: a new display, and the compositor's socket path, which it copies.
    let display = unsafe { (api.wpe_display_wayland_new)() };
    let mut error = std::ptr::null_mut();
    let connected =
        unsafe { (api.wpe_display_wayland_connect)(display, socket.as_ptr(), &mut error) } != 0;
    // SAFETY: the error the call above may have set.
    if let Some((_, why)) = unsafe { wpe::take_error(api, error) } {
        return Err(format!(
            "WPE could not connect to the web view compositor: {why}"
        ));
    }
    if !connected {
        return Err("WPE could not connect to the web view compositor".to_string());
    }
    let identifier = foundation::bundle::main()
        .and_then(|bundle| bundle.identifier())
        .unwrap_or("rosemary")
        .to_string();
    let home = foundation::host_home_directory();
    let data = home.join("Library/WebKit").join(&identifier);
    let cache = home.join("Library/Caches").join(&identifier).join("WebKit");
    let data = wpe::c_string(&data.to_string_lossy());
    let cache = wpe::c_string(&cache.to_string_lossy());
    // SAFETY: two NUL-terminated paths, which the session copies.
    let session = unsafe { (api.webkit_network_session_new)(data.as_ptr(), cache.as_ptr()) };
    if session.is_null() {
        return Err("WPE WebKit made no network session".to_string());
    }
    appkit::event::add_pump_hook(pump);
    Ok(Engine {
        api,
        compositor,
        display: display as usize,
        session: session as usize,
        thread: darwin::pthread::id(),
    })
}

/// Get the main thread to the pump, from the compositor's thread.
fn wake() {
    corefoundation::runloop::main().wake();
}

pub fn warn_once(message: &str) {
    static WARNED: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    let mut warned = WARNED.lock().unwrap_or_else(|e| e.into_inner());
    if !warned.iter().any(|seen| seen == message) {
        eprintln!("rosemary: webkit: {message}");
        warned.push(message.to_string());
    }
}

pub fn tracing() -> bool {
    static TRACE: OnceLock<bool> = OnceLock::new();
    *TRACE.get_or_init(|| std::env::var_os("ROSEMARY_TRACE_WEBKIT").is_some())
}

thread_local! {
    static DEFERRED: RefCell<Vec<Box<dyn FnOnce()>>> = const { RefCell::new(Vec::new()) };
}

/// Run `work` at the next turn of the pump.
///
/// For a completion handler WebKit calls asynchronously: calling one before
/// the method that took it has returned is a reentrancy its caller does not
/// expect.
pub fn defer(work: impl FnOnce() + 'static) {
    DEFERRED.with(|deferred| deferred.borrow_mut().push(Box::new(work)));
}

fn pump() {
    let Some(engine) = running() else {
        return;
    };
    let pool = objc::arc::pool_push();
    // Bounded, as `drain_main` is: a context that always has work would
    // otherwise keep the run loop from ever getting back to input and drawing.
    for _ in 0..64 {
        // SAFETY: the default context, iterated on the thread WPE runs on.
        if unsafe { (engine.api.g_main_context_iteration)(std::ptr::null_mut(), 0) } == 0 {
            break;
        }
    }
    let work = DEFERRED.with(|deferred| std::mem::take(&mut *deferred.borrow_mut()));
    for item in work {
        item();
    }
    crate::web_view::tick(engine);
    restore_guest_gl();
    objc::arc::pool_pop(pool);
}

/// WPE binds EGL directly, bypassing SDL's TLS. Unbind before restoring the
/// guest's binding; otherwise SDL may skip the native bind because its cached
/// pointers match.
fn restore_guest_gl() {
    let context = sdl3::video::gl_current_context();
    let window = sdl3::video::gl_current_window();
    if !context.is_null() && !sdl3::video::egl::backs(context) {
        sdl3::video::gl_make_current(std::ptr::null_mut(), std::ptr::null_mut());
        if !sdl3::video::gl_make_current(window, context) {
            warn_once("could not restore the guest OpenGL context after web rendering");
        }
    }
}

/// A guest block kept past the call that passed it.
pub struct Block(usize);

impl Block {
    pub fn copy(block: Id) -> Option<Block> {
        (!block.is_null()).then(|| Block(objc::block::copy(block) as usize))
    }

    fn invoke(&self) -> usize {
        // SAFETY: a block literal, whose `invoke` is its third word.
        unsafe { *((self.0 as *const u8).add(BLOCK_INVOKE_OFFSET) as *const usize) }
    }

    /// Call a `void (^)(id)`.
    pub fn call1(&self, first: Id) {
        let invoke = self.invoke();
        if invoke == 0 {
            return;
        }
        // SAFETY: the caller took this block for a handler of this prototype.
        unsafe {
            let f: unsafe extern "C" fn(Id, Id) = std::mem::transmute(invoke);
            f(self.0 as Id, first);
        }
    }

    /// Call a `void (^)(id, id)`.
    pub fn call2(&self, first: Id, second: Id) {
        let invoke = self.invoke();
        if invoke == 0 {
            return;
        }
        unsafe {
            let f: unsafe extern "C" fn(Id, Id, Id) = std::mem::transmute(invoke);
            f(self.0 as Id, first, second);
        }
    }

    /// Call a `void (^)(void)`.
    pub fn call0(&self) {
        let invoke = self.invoke();
        if invoke == 0 {
            return;
        }
        unsafe {
            let f: unsafe extern "C" fn(Id) = std::mem::transmute(invoke);
            f(self.0 as Id);
        }
    }
}

impl Drop for Block {
    fn drop(&mut self) {
        objc::block::release(self.0 as Id);
    }
}

/// `isa`, `flags`, `reserved`, then `invoke`.
const BLOCK_INVOKE_OFFSET: usize = 16;

#[repr(C)]
struct Descriptor {
    reserved: u64,
    size: u64,
}

/// A block made here for a guest to call, carrying `context` after the
/// literal's fields.
#[repr(C)]
pub struct HostBlock<T> {
    isa: *const c_void,
    flags: i32,
    reserved: i32,
    invoke: *const c_void,
    descriptor: *const Descriptor,
    pub context: T,
}

/// A block whose `invoke` is `invoke`, which receives the block itself first.
///
/// Marked global, so a guest's copy of it is the block itself and a release
/// does nothing. It is leaked: a guest may hold it, and release it, after it has
/// been called, and a few words per navigation is the price of never freeing
/// it under someone.
pub fn host_block<T>(invoke: *const c_void, context: T) -> Id {
    let isa = objc::class::lookup(c"__NSGlobalBlock__")
        .map(|class| objc::class::ptr(class) as *const c_void)
        .expect("registered by `objc::block`");
    let descriptor = Box::leak(Box::new(Descriptor {
        reserved: 0,
        size: size_of::<HostBlock<T>>() as u64,
    }));
    Box::leak(Box::new(HostBlock {
        isa,
        // `BLOCK_IS_GLOBAL`.
        flags: 1 << 28,
        reserved: 0,
        invoke,
        descriptor,
        context,
    })) as *mut HostBlock<T> as Id
}

/// Whether `object` answers `selector`, asked the way a caller would ask, so
/// that a proxy's forwarding is honoured.
pub fn responds(object: Id, selector: &str) -> bool {
    if object.is_null() {
        return false;
    }
    // SAFETY: a live object; `-respondsToSelector:` is `BOOL (id, SEL, SEL)`.
    unsafe {
        let send: unsafe extern "C" fn(Id, objc::sel::Sel, objc::sel::Sel) -> u8 =
            std::mem::transmute(objc::msg::entry());
        send(
            object,
            objc::sel::get("respondsToSelector:"),
            objc::sel::get(selector),
        ) != 0
    }
}

/// An autoreleased string.
pub fn string(text: &str) -> Id {
    objc::arc::autorelease(foundation::string::from_str(text))
}

/// A string object's text, or empty for nil.
pub fn text_of(string: Id) -> String {
    if string.is_null() {
        return String::new();
    }
    // SAFETY: null or a string object, which every caller passes.
    String::from_utf8_lossy(&unsafe { foundation::string::bytes(string) }).into_owned()
}

/// Release a GObject reference held as an address.
pub fn unref(api: &Api, object: usize) {
    if object != 0 {
        // SAFETY: a reference the caller owns.
        unsafe { (api.g_object_unref)(object as Ptr) };
    }
}

#[cfg(all(test, target_os = "linux"))]
mod gl_tests {
    use super::*;

    #[test]
    #[ignore = "requires a host EGL driver"]
    fn web_rendering_restores_native_gl_even_when_sdl_cached_state_matches() {
        appkit::bootstrap();
        let window = sdl3::video::create_window(
            c"GL restoration test",
            2,
            2,
            sdl3::video::flags::OPENGL | sdl3::video::flags::HIDDEN,
        )
        .expect("GL window");
        let guest = sdl3::video::gl_create_context(window).expect("guest context");
        let foreign = sdl3::video::gl_create_context(window).expect("foreign context");
        assert!(sdl3::video::egl::backs(foreign), "test needs EGL");
        assert!(sdl3::video::gl_make_current(window, guest));
        let symbol = |name: &std::ffi::CStr| {
            // SAFETY: a static symbol name in the EGL driver already loaded by SDL.
            let address = unsafe { libc::dlsym(libc::RTLD_DEFAULT, name.as_ptr()) };
            assert!(!address.is_null(), "missing {name:?}");
            address
        };
        // SAFETY: checked EGL entry points, with their declared prototypes.
        unsafe {
            let current_display: unsafe extern "C" fn() -> Ptr =
                std::mem::transmute(symbol(c"eglGetCurrentDisplay"));
            let current_surface: unsafe extern "C" fn(u32) -> Ptr =
                std::mem::transmute(symbol(c"eglGetCurrentSurface"));
            let make_current: unsafe extern "C" fn(Ptr, Ptr, Ptr, Ptr) -> u32 =
                std::mem::transmute(symbol(c"eglMakeCurrent"));
            const EGL_DRAW: u32 = 0x3059;
            const EGL_READ: u32 = 0x305A;
            assert_ne!(
                make_current(
                    current_display(),
                    current_surface(EGL_DRAW),
                    current_surface(EGL_READ),
                    foreign
                ),
                0
            );
        }
        assert_eq!(sdl3::video::gl_current_context(), guest);
        assert!(sdl3::video::egl::backs(foreign));
        restore_guest_gl();
        assert_eq!(sdl3::video::gl_current_context(), guest);
        assert_eq!(sdl3::video::gl_current_window(), window);
        assert!(sdl3::video::egl::backs(guest));
        sdl3::video::gl_make_current(std::ptr::null_mut(), std::ptr::null_mut());
        sdl3::video::gl_destroy_context(foreign);
        sdl3::video::gl_destroy_context(guest);
        sdl3::video::destroy_window(window);
    }
}
