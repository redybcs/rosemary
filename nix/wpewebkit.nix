{ lib
, clangStdenv
, fetchurl
, replaceVars
, addDriverRunpath
, bison
, cmake
, gettext
, gperf
, ninja
, perl
, pkg-config
, python3
, ruby
, glib
, unifdef
, wayland-scanner
, at-spi2-core
, enchant
, expat
, flite
, fontconfig
, freetype
, harfbuzzFull
, hyphen
, icu
, lcms2
, libavif
, libbacktrace
, libdrm
, libepoxy
, libgbm
, libgcrypt
, libgpg-error
, libjpeg
, libjxl
, libmanette
, libpng
, libseccomp
, libsoup_3
, libsysprof-capture
, libtasn1
, libwebp
, libxkbcommon
, libxml2
, libxslt
, nettle
, sqlite
, systemdLibs
, wayland
, wayland-protocols
, woff2
, zlib
, gst_all_1
, bubblewrap
, xdg-dbus-proxy
}:

# WPE WebKit with only the WPEPlatform (2.0) API and its Wayland backend:
# crates/webkit embeds its views in a private Wayland compositor and never
# uses libwpe, DRM or headless output. nixpkgs packages only WebKitGTK.
clangStdenv.mkDerivation (finalAttrs: {
  pname = "wpewebkit";
  version = "2.54.3";

  outputs = [ "out" "dev" ];

  # Can't be linked within a 4GB address space (nixpkgs#153528).
  separateDebugInfo = true;

  src = fetchurl {
    url = "https://wpewebkit.org/releases/wpewebkit-${finalAttrs.version}.tar.xz";
    hash = "sha256-eHr9Mq4j6sBhYhRlOQ9EpGLodKnrq6C2VSHXI3+opeQ=";
  };

  # nixpkgs' WebKitGTK patch: the web process sandbox must see the store
  # and the graphics driver.
  patches = [
    (replaceVars ./wpewebkit/fix-bubblewrap-paths.patch {
      inherit (builtins) storeDir;
      inherit (addDriverRunpath) driverLink;
    })
  ];

  postPatch = ''
    patchShebangs .
  '';

  nativeBuildInputs = [
    bison
    cmake
    gettext
    gperf
    ninja
    perl
    pkg-config
    python3
    ruby
    glib # gdbus-codegen
    unifdef
    wayland-scanner
  ];

  buildInputs = [
    at-spi2-core
    enchant
    expat
    flite
    fontconfig
    freetype
    harfbuzzFull
    hyphen
    icu
    lcms2
    libavif
    libbacktrace
    libdrm
    libepoxy
    libgbm
    libgcrypt
    libgpg-error
    libjpeg
    libjxl
    libmanette
    libpng
    libseccomp
    libsysprof-capture
    libtasn1
    libwebp
    libxkbcommon
    libxml2
    libxslt
    nettle
    sqlite
    systemdLibs
    wayland
    wayland-protocols
    woff2
    zlib
    gst_all_1.gst-plugins-base
    gst_all_1.gst-plugins-bad
  ];

  propagatedBuildInputs = [ glib libsoup_3 ];

  cmakeFlags = [
    "-DPORT=WPE"
    "-DENABLE_WPE_LEGACY_API=OFF"
    "-DENABLE_WPE_PLATFORM_DRM=OFF"
    "-DENABLE_WPE_PLATFORM_HEADLESS=OFF"
    "-DENABLE_WPE_QT_API=OFF"
    "-DENABLE_INTROSPECTION=OFF"
    "-DENABLE_DOCUMENTATION=OFF"
    "-DENABLE_MINIBROWSER=OFF"
    "-DBWRAP_EXECUTABLE=${lib.getExe bubblewrap}"
    "-DDBUS_PROXY_EXECUTABLE=${lib.getExe xdg-dbus-proxy}"
  ];

  requiredSystemFeatures = [ "big-parallel" ];

  meta = {
    description = "Web content rendering engine, WPE port";
    homepage = "https://wpewebkit.org/";
    license = lib.licenses.bsd2;
    platforms = lib.platforms.linux;
  };
})
