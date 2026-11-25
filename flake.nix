{
  description = "Run dumped Xbox Series GDK titles under WineGDK + vkd3d-proton with AnyPS5-translated shaders";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    # Archived 2026-11-13; successor is xodus-gaming/{wine,xgameruntime}, but
    # that runtime is still a skeleton, so WineGDK remains the most complete.
    winegdk-src = {
      url = "github:Weather-OS/WineGDK";
      flake = false;
    };
    # The Xodus project's xgameruntime IDLs (LGPL-2.1): shims/xgameruntime's interface table is
    # generated from them. Pinned to the commit americas-recomp's CI builds its runtime against.
    xodus-xgameruntime = {
      url = "github:xodus-gaming/xgameruntime/642e41c7617310c5d1b39d4e525b01ecae261f9e";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, winegdk-src, xodus-xgameruntime }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      lib = pkgs.lib;
      mingwCC = pkgs.pkgsCross.mingwW64.stdenv.cc;
      python = pkgs.python3.withPackages (p: [ p.pefile p.capstone ]);

      # The shims' sources, also given to rebox-run, which rebuilds them in the prefix with
      # stubs for imports a title needs and nothing provides.
      shims-src = lib.fileset.toSource {
        root = ./.;
        fileset = lib.fileset.unions [
          ./Makefile
          ./shims
          ./anyps5/rebox_anyps5.h
          ./layers/gpuva_mailbox.h
          (lib.fileset.maybeMissing ./generated)
        ];
      };
      # The variables nix's setup hooks give mingw's compiler wrapper inside a build (its role,
      # and the mcfgthread include and library paths), for rebox-run's rebuild outside one.
      mingwBuildEnv = pkgs.stdenvNoCC.mkDerivation {
        name = "mingw-build-env";
        nativeBuildInputs = [ mingwCC ];
        dontUnpack = true;
        installPhase = ''
          { printf 'export NIX_CFLAGS_COMPILE=%q\n' "$NIX_CFLAGS_COMPILE"
            printf 'export NIX_LDFLAGS=%q\n' "$NIX_LDFLAGS"
            env | grep '^NIX_.*_WRAPPER_TARGET_HOST_' | sed 's/^/export /'; } > $out
        '';
      };
      shimsMakeFlags = [ "MINGW_CC=x86_64-w64-mingw32-gcc" "MINGW_CXX=x86_64-w64-mingw32-g++" "FFMPEG_XMA2=${self.packages.${system}.ffmpeg-xma2}"
                         "XODUS_IDL=${xodus-xgameruntime}/include" "ADDRLIB=${pkgs.mesa.src}/src/amd/addrlib" ];

      pyTool = name: file: pkgs.runCommand name { nativeBuildInputs = [ pkgs.makeWrapper ]; } ''
        makeWrapper ${python}/bin/python3 $out/bin/${name} --add-flags ${file} \
          --suffix PATH : ${mingwCC}/bin
      '';
    in
    {
      packages.${system} = rec {
        # gstreamerSupport: winegstreamer, which Media Foundation's H.264 and AAC decoders are
        # (Unreal 5's Electra player creates them; without it they aren't registered).
        winegdk = (pkgs.wineWow64Packages.unstable.override { gstreamerSupport = true; }).overrideAttrs (old: {
          pname = "winegdk";
          version = "11.14-gdk-${winegdk-src.shortRev}";
          src = winegdk-src;
          # Fixes Xbox titles need from Wine itself (patches/wine/).
          patches = (old.patches or [ ]) ++ [ ./patches/wine/0001-ntdll-accept-console-large-page-flags.patch ];
        });

        # The nixpkgs vkd3d-proton package is the native Linux build; Wine
        # needs the PE DLLs from the release tarball.
        vkd3d-proton-win = pkgs.stdenvNoCC.mkDerivation rec {
          pname = "vkd3d-proton-win";
          version = "3.0.1";
          src = pkgs.fetchurl {
            url = "https://github.com/HansKristian-Work/vkd3d-proton/releases/download/v${version}/vkd3d-proton-${version}.tar.zst";
            hash = "sha256-PPIxVSKvXkNgXvbTxB2tkThwQL+XGZk08/erdsqqLww=";
          };
          nativeBuildInputs = [ pkgs.zstd ];
          installPhase = "mkdir -p $out && cp -r x64 x86 $out/";
        };

        # RADV that honours requested device addresses in the lower half of
        # the VA space, so GPU-visible console memory can sit at its CPU
        # address (docs/d3d12x.md). Only the drivers this project needs.
        # Use with VK_ICD_FILENAMES=$out/share/vulkan/icd.d/radeon_icd.x86_64.json.
        # Host Vulkan layer that maps host-visible memory at identical GPU
        # and CPU addresses (layers/gpuva_identity.c).
        # Static mingw libavcodec/libavutil/libswresample with only FFmpeg's XMA2
        # and Opus decoders, linked into shims/xaudio2_9 (XMA2) and shims/xapu
        # (Opus) (nixpkgs' mingw FFmpeg is marked broken).
        ffmpeg-xma2 = pkgs.stdenv.mkDerivation {
          pname = "ffmpeg-xma2-mingw";
          inherit (pkgs.ffmpeg) version src;
          nativeBuildInputs = [ mingwCC pkgs.nasm pkgs.pkg-config pkgs.perl ];
          dontAddPrefix = true;
          dontStrip = true; # native strip on mingw archives
          configurePhase = ''
            ./configure --prefix=$out --target-os=mingw32 --arch=x86_64 --enable-cross-compile \
              --cross-prefix=x86_64-w64-mingw32- --disable-everything --enable-decoder=xma2 --enable-decoder=opus \
              --disable-programs --disable-doc --disable-avdevice --disable-avformat --disable-swscale \
              --disable-avfilter --disable-network --disable-debug \
              --enable-static --disable-shared --disable-autodetect --disable-asm
          '';
        };

        gpuva-layer = pkgs.stdenv.mkDerivation {
          name = "rebox-gpuva-layer";
          src = lib.fileset.toSource { root = ./.; fileset = ./layers; };
          buildInputs = [ pkgs.vulkan-headers ];
          buildPhase = ''
            $CC -O2 -g -Wall -fPIC -shared -fvisibility=hidden \
              -o librebox_gpuva_identity.so layers/gpuva_identity.c -lpthread
          '';
          installPhase = ''
            mkdir -p $out/lib $out/share/vulkan/explicit_layer.d
            cp librebox_gpuva_identity.so $out/lib/
            sed "s|./librebox_gpuva_identity.so|$out/lib/librebox_gpuva_identity.so|" \
              layers/VK_LAYER_REBOX_gpuva_identity.json \
              > $out/share/vulkan/explicit_layer.d/VK_LAYER_REBOX_gpuva_identity.json
          '';
        };

        # Valve's Fossilize: its layer records every pipeline a run creates, and fossilize-replay
        # builds them again with no title running, filling the driver's disk cache (bin/rebox-run,
        # REBOX_FOSSILIZE_DB). Not in nixpkgs; its own manifest names the library by a doubled
        # store path, so the layer gets a manifest of ours.
        fossilize = pkgs.stdenv.mkDerivation {
          pname = "fossilize";
          version = "556da9f";
          src = pkgs.fetchFromGitHub {
            owner = "ValveSoftware";
            repo = "Fossilize";
            rev = "556da9ff7d7616704dda515756219ab1dab693ec";
            fetchSubmodules = true;
            hash = "sha256-7u/aepE5YGbEy5nF9PnxDr+ljNmrE/6iwS/z8cdIcBM=";
          };
          nativeBuildInputs = [ pkgs.cmake pkgs.python3 pkgs.makeWrapper ];
          cmakeFlags = [ "-DFOSSILIZE_VULKAN_LAYER=ON" "-DFOSSILIZE_TESTS=OFF" "-DCMAKE_BUILD_TYPE=Release" ];
          # The tools load libvulkan at run time (volk): without the loader on their path every
          # replay worker failed at once ("volkInitialize failed").
          postInstall = ''
            for f in $out/bin/*; do
              wrapProgram "$f" --prefix LD_LIBRARY_PATH : ${pkgs.vulkan-loader}/lib
            done
            mkdir -p $out/share/vulkan/rebox_layer.d
            sed "s|\"library_path\": \".*\"|\"library_path\": \"$out/lib/libVkLayer_fossilize.so\"|" \
              $out/share/vulkan/implicit_layer.d/VkLayer_fossilize.json \
              > $out/share/vulkan/rebox_layer.d/VkLayer_fossilize.json
          '';
        };

        radv-rebox = (pkgs.mesa.override {
          galliumDrivers = [ "llvmpipe" "radeonsi" ];
          vulkanDrivers = [ "amd" "swrast" ];
          vulkanLayers = [ ];
        }).overrideAttrs (old: {
          pname = "radv-rebox";
          patches = (old.patches or [ ]) ++ [ ./patches/mesa-radv-low-va-replay.patch ./patches/mesa-radv-low-32bit-window.patch ./patches/mesa-radv-low-va-no-pad.patch ./patches/mesa-radv-fault-report.patch ];
        });

        # vkd3d-proton's PE DLLs built from source with our patches
        # (patches/vkd3d-proton/*.patch, applied in name order), for console
        # shaders: docs/shaders.md. Same source and version as nixpkgs'
        # native package; cross-compiled with its own build-win64.txt.
        vkd3d-proton-rebox = pkgs.stdenvNoCC.mkDerivation {
          pname = "vkd3d-proton-rebox";
          inherit (pkgs.vkd3d-proton) version src;
          patches = lib.sort lib.lessThan (lib.filter (lib.hasSuffix ".patch")
            (lib.filesystem.listFilesRecursive ./patches/vkd3d-proton));
          nativeBuildInputs = [ mingwCC pkgs.meson pkgs.ninja pkgs.glslang pkgs.python3 pkgs.wineWow64Packages.unstable ];
          postPatch = ''
            substituteInPlace meson.build \
              --replace-fail "vkd3d_build = vcs_tag(" \
                             "vkd3d_build = vcs_tag( fallback : '$(cat .nixpkgs-auxfiles/vkd3d_build)'", \
              --replace-fail "vkd3d_version = vcs_tag(" \
                             "vkd3d_version = vcs_tag( fallback : '$(cat .nixpkgs-auxfiles/vkd3d_version)'",
          '';
          configurePhase = ''
            # -static links winpthreads (vkd3d-proton uses pthreads on Windows
            # too); cross gcc ignores LIBRARY_PATH, so pass it to meson.
            pthreads=${pkgs.pkgsCross.mingwW64.windows.pthreads}/lib
            # enable_trace: breadcrumbs (VKD3D_CONFIG=breadcrumbs) name the commands a lost device was
            # running; trace messages stay off unless WINEDEBUG asks for them.
            meson setup build --cross-file build-win64.txt --buildtype release --prefix $out -Denable_trace=true \
              -Dc_link_args=-L$pthreads -Dcpp_link_args=-L$pthreads
          '';
          buildPhase = "ninja -C build -j$NIX_BUILD_CORES";
          installPhase = ''
            mkdir -p $out/x64
            cp build/libs/d3d12/d3d12.dll build/libs/d3d12core/d3d12core.dll $out/x64/
          '';
          dontStrip = true; # native strip on PE files
        };

        # AnyPS5's RDNA -> SPIR-V recompiler as rebox_anyps5.dll (anyps5/,
        # docs/anyps5.md): a sparse checkout of AnyPS5 at a pinned commit, its
        # three shader submodules at the revisions AnyPS5 pins, and
        # patches/anyps5/ applied.
        anyps5-src = let
          sparse = pkgs.fetchgit {
            url = "https://github.com/boykopovar/AnyPS5";
            rev = "fc973e01897731a88e879fbd801e4ef5c6c29643";
            sparseCheckout = [ "core/shader/recompiler" "core/libs/prx/libc/include" ];
            hash = "sha256-wSwfEePRhd3G63sX7P6q1XvO/yko2iDwK4nYS9Rvk5U=";
          };
          sub = repo: rev: hash: pkgs.fetchFromGitHub { owner = "KhronosGroup"; inherit repo rev hash; };
          headers = sub "SPIRV-Headers" "b824a462d4256d720bebb40e78b9eb8f78bbb305" "sha256-HjJjMuqTrYv5LUOWcexzPHb8nhOT4duooDAhDsd44Zo=";
          tools = sub "SPIRV-Tools" "262bdab48146c937467f826699a40da0fdfc0f1a" "sha256-H+t7ZH4SB+XgWTLj9XaJWZwAWk8M2QeC98Zi5ay8PBc=";
          glslang = sub "glslang" "ba1640446f3826a518721d1f083f3a8cca1120c3" "sha256-xplVnLcfVTg/iUScla8TAdHzvI/jq+cjLUSRgKNBKDI=";
        in pkgs.runCommand "anyps5-src" { nativeBuildInputs = [ pkgs.git ]; } ''
          cp -r ${sparse} $out && chmod -R u+w $out
          mkdir -p $out/3rdparty
          cp -r ${headers} $out/3rdparty/SPIRV-Headers
          cp -r ${tools} $out/3rdparty/SPIRV-Tools
          cp -r ${glslang} $out/3rdparty/glslang
          chmod -R u+w $out
          # Only the recompiler is built: skip hunks for AnyPS5's driver, tests and docs.
          for p in ${./patches/anyps5}/*.patch; do git -C $out apply --include='core/shader/*' "$p"; done
        '';

        # With mimalloc (MIT) linked in for the DLL's new and delete (anyps5/allocator.cpp).
        rebox-anyps5 = pkgs.stdenvNoCC.mkDerivation {
          name = "rebox-anyps5";
          src = lib.fileset.toSource { root = ./.; fileset = ./anyps5; };
          nativeBuildInputs = [ mingwCC pkgs.cmake pkgs.ninja pkgs.python3 ];
          dontUseCmakeConfigure = true;
          buildPhase = ''
            cmake -S anyps5 -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
              -DANYPS5_SOURCE=${anyps5-src} -DCMAKE_TOOLCHAIN_FILE=$PWD/anyps5/mingw-w64.cmake \
              -DMINGW_PTHREADS=${pkgs.pkgsCross.mingwW64.windows.pthreads} -DMIMALLOC_SOURCE=${pkgs.mimalloc.src}
            cmake --build build --target rebox_anyps5 sa5_check -j$NIX_BUILD_CORES
          '';
          installPhase = "mkdir -p $out && cp build/rebox_anyps5.dll build/sa5_check.exe $out/";
          dontStrip = true;
        };

        # Only dxgi.dll is used: vkd3d-proton needs a DXGI that enumerates
        # Vulkan adapters without a display, as in Proton.
        dxvk-win = pkgs.stdenvNoCC.mkDerivation rec {
          pname = "dxvk-win";
          version = "3.1.1";
          src = pkgs.fetchurl {
            url = "https://github.com/doitsujin/dxvk/releases/download/v${version}/dxvk-${version}.tar.gz";
            hash = "sha256-QFZbSnJKrcRDP6TgELSyORbZsfG67uZOFxhtuU9U5gg=";
          };
          installPhase = "mkdir -p $out && cp -r x64 x32 $out/";
        };

        shims = pkgs.stdenvNoCC.mkDerivation {
          name = "rebox-shims";
          src = shims-src;
          nativeBuildInputs = [ mingwCC pkgs.python3 ];
          makeFlags = shimsMakeFlags;
          buildFlags = [ "shims" ];
          installPhase = ''
            mkdir -p $out
            cp build/*.dll $out/
            if [ -d build/stub ]; then mkdir -p $out/stub && cp build/stub/*.dll $out/stub/; fi
            cp ${rebox-anyps5}/rebox_anyps5.dll $out/
          '';
        };

        scan-imports = pyTool "scan_imports.py" ./tools/scan_imports.py;
        # mingw gcc on PATH lets the probe find d3d12.h for GUID and vtable names.
        d3d12x-probe = pyTool "d3d12x_probe.py" ./tools/d3d12x_probe.py;

        rebox-run = pkgs.writeShellScriptBin "rebox-run" ''
          export PATH=${lib.makeBinPath [ pkgs.coreutils pkgs.gnugrep pkgs.gnused ]}:$PATH
          export WINE=''${WINE:-${winegdk}/bin/wine}
          export VKD3D_PROTON_DIR=''${VKD3D_PROTON_DIR:-${vkd3d-proton-rebox}/x64}
          export DXVK_DIR=''${DXVK_DIR:-${dxvk-win}/x64}
          export REBOX_SHIMS_DIR=''${REBOX_SHIMS_DIR:-${shims}}
          export REBOX_SCAN=${scan-imports}/bin/scan_imports.py
          export REBOX_SRC=${shims-src}
          export REBOX_MAKE_FLAGS="${lib.concatStringsSep " " shimsMakeFlags}"
          export REBOX_BUILD_ENV=${mingwBuildEnv}
          export REBOX_CA_BUNDLE=''${REBOX_CA_BUNDLE:-${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt}
          export PATH=$PATH:${lib.makeBinPath [ mingwCC pkgs.gnumake pkgs.python3 pkgs.findutils ]}
          export REBOX_GPUVA_LAYER_DIR=''${REBOX_GPUVA_LAYER_DIR:-${gpuva-layer}/share/vulkan/explicit_layer.d}
          export REBOX_RADV_ICD=''${REBOX_RADV_ICD:-${radv-rebox}/share/vulkan/icd.d/radeon_icd.x86_64.json}
          export REBOX_FOSSILIZE_DIR=''${REBOX_FOSSILIZE_DIR:-${fossilize}}
          exec ${pkgs.bash}/bin/bash ${./bin/rebox-run} "$@"
        '';

        default = rebox-run;
      };

      apps.${system} = {
        default = { type = "app"; program = "${self.packages.${system}.rebox-run}/bin/rebox-run"; };
        scan-imports = { type = "app"; program = "${self.packages.${system}.scan-imports}/bin/scan_imports.py"; };
        d3d12x-probe = { type = "app"; program = "${self.packages.${system}.d3d12x-probe}/bin/d3d12x_probe.py"; };
      };

      # Stock Wine here so the shell doesn't need a WineGDK build; the shims
      # and the smoke test don't depend on WineGDK.
      devShells.${system}.default = pkgs.mkShell {
        packages = [ mingwCC python pkgs.gnumake pkgs.wineWow64Packages.unstable pkgs.vkd3d pkgs.vulkan-tools pkgs.vulkan-headers pkgs.vulkan-loader pkgs.directx-shader-compiler pkgs.glslang ];
        shellHook = ''
          export WINE=wine
          export VKD3D_PROTON_DIR=${self.packages.${system}.vkd3d-proton-rebox}/x64
          export DXVK_DIR=${self.packages.${system}.dxvk-win}/x64
          export REBOX_SHIMS_DIR=$PWD/build
          export FFMPEG_XMA2=${self.packages.${system}.ffmpeg-xma2}
          export XODUS_IDL=${xodus-xgameruntime}/include
          # AMD's addrlib (Mesa, MIT), for detiling console textures in d3d12_xs.
          export ADDRLIB=${pkgs.mesa.src}/src/amd/addrlib
          export REBOX_ANYPS5_DIR=${self.packages.${system}.rebox-anyps5}
          export WINE_DLL_DIR=${pkgs.wineWow64Packages.unstable}/lib/wine/x86_64-windows
          # Software Vulkan only for machines (or sandboxes) without a GPU;
          # otherwise the system's drivers (e.g. RADV) are used.
          if [ -z "''${VK_ICD_FILENAMES:-}" ] && ! ls /dev/dri/renderD* >/dev/null 2>&1; then
            export VK_ICD_FILENAMES=${pkgs.mesa}/share/vulkan/icd.d/lvp_icd.x86_64.json
          fi
        '';
      };
    };
}
