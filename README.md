<p align="center"><img src="logo.png" alt="Rebox" width="480"></p>

# Rebox

Rebox runs dumped Xbox Series (Scarlett) GDK titles on Linux. It pairs WineGDK with
vkd3d-proton for D3D12, shims for the console-only DLLs and D3D12.X interfaces, and the AnyPS5
recompiler, which turns the console's RDNA shaders into SPIR-V at dispatch time. It is
work in progress: titles start and run their game code, but rendering is incomplete (see
[Known gaps](#known-gaps-ordered-roughly-by-how-soon-theyll-bite)).

Rebox targets the same titles as other Xbox-on-PC projects. It runs a GDK title to gameplay and
an Unreal Engine 5 title to its title screen, and is bringing up an Unreal Engine 5 sample.

### Why AnyPS5

Rebox compiles the console's RDNA shaders straight to SPIR-V with AnyPS5, at a shader's first
dispatch or draw, rather than lifting them to HLSL and compiling that with DXC. There is no DXC
step: other projects report single shaders taking minutes in DXC, while AnyPS5 builds pipelines in
milliseconds, and a shader keeps one module however its descriptors move. The cost is that AnyPS5
has to model the hardware closely (user SGPRs, descriptor formats, wave size). Titles render
frames through AnyPS5-compiled shaders today; frame rates haven't been compared with other projects
yet.

It is built from:

- **WineGDK** for Wine itself. Its Gaming Runtime (`xgameruntime`) answers only five API families
  and its XUser is stubs, so `shims/xgameruntime` replaces it: americas-recomp's stand-in, with an
  offline local user, local saves, persistent local storage, the title as one fully installed
  package, an active store licence and complete task queues ([its README](shims/xgameruntime/README.md)).
  WineGDK's GameInput reads only mice, so `shims/gameinput` replaces it: GameInput v0 (the
  console's in-box version) with one gamepad from XInput, or from the keyboard when no controller
  is connected.
  WineGDK comes with `patches/wine/`: `VirtualAlloc` accepts the `MEM_LARGE_PAGES` and `MEM_4MB_PAGES` flags Xbox
  titles pass (ordinary pages back them). `rebox-run` also sets the `ConsoleMode` registry value
  the GDK's `XSystemGetDeviceType` reads, so the title knows it's on a Series X.
- **vkd3d-proton** for D3D12 on Vulkan, built from source with our patches (`.#vkd3d-proton-rebox`,
  `patches/vkd3d-proton/`), plus **DXVK's `dxgi.dll`** for adapter enumeration
- **shims for the console-only DLLs** WineGDK doesn't cover: `xmem.dll` and `d3d12_xs.dll` (D3D12.X)
- **AnyPS5's RDNA → SPIR-V recompiler** for the console shaders (`rebox_anyps5.dll`), run at
  each dispatch against console-format shadow descriptor heaps ([docs/anyps5.md](docs/anyps5.md));
  compute and graphics stages compile at dispatch and draw time, on several threads, with
  **mimalloc** (MIT) behind the DLL's `new` and `delete`

A title is a dumped package: loose files plus `MicrosoftGame.Config` and the main `.exe`.
Development is on a Radeon 890M (RADV). Audio is silent: the audio coprocessor answers but
mixes nothing.

## Layout

```
bin/rebox-run          launcher: prefix setup, DLL install/overrides, runs the exe from MicrosoftGame.Config
shims/xmem/               xmem.dll: XMemAlloc/Free(+Default), allocation hooks, XMemVirtualQuery;
                          GPU-visible memory types come from d3d12_xs
shims/d3d12_xs/           d3d12_xs.dll: Xbox device/queue/command-list/descriptor-heap proxies (proxy.c),
                          GPU memory (gpu_memory.c), shader substitution, forwards to vkd3d-proton
shims/amd_fidelityfx_loader_xbox/  FidelityFX API stub: FSR-style queries, bilinear upscale via d3d12_xs
shims/xaudio2_9/          xaudio2_9.dll: Wine's XAudio2 (via xaudio2_8) with XMA2 voices played as silence
shims/acphal/             acphal.dll: the APU heap, and an audio coprocessor channel that acknowledges
                          commands and ticks audio frames (nothing is mixed)
shims/xapu/               xapu.dll: the audio decoder queue: Opus packets decoded by FFmpeg as each command is queued
shims/chad/               chad.dll: the audio DSP (XDsp); it connects and takes commands that do nothing
shims/xcrdapi/            XCRDAPI.dll: DirectStorage queues, zlib, and BCPack (bcpack.c) decoded in process
layers/                   VK_LAYER_REBOX_gpuva_identity: host Vulkan layer mapping host-visible
                          memory at the same GPU and CPU address
patches/                  Mesa patch letting RADV place memory at lower-half device addresses;
                          vkd3d-proton, AnyPS5 and Wine patches
shims/common/             logging + REBOX_STUB macro for generated stubs
generated/                stubs from tools/scan_imports.py --gen-stubs (one-off fill-in exports;
                          dlls/<name>/ for whole DLLs nobody provides) and generated shaders
tools/scan_imports.py     which imports can't be resolved under Wine + shims; generates stubs
tools/d3d12x_probe.py     recovers D3D12.X IIDs, vtable slots and inlined field accesses from console binaries
tests/                    smoke test (xmem, forwarders, PSO substitution, negative control);
                          bcpack_test.c tests the BCPack decoder natively;
                          gpuva_probe.c checks whether the Vulkan driver can map GPU VA == CPU VA
docs/d3d12x.md            D3D12.X interfaces, unified memory and presentation
docs/anyps5.md            console shaders through AnyPS5: shadow heaps, compute and graphics
docs/shaders.md           the vkd3d-proton patches and the shader cache
```

## Usage

```sh
nix run . -- --scan path/to/title    # report imports Wine + vkd3d-proton + shims can't resolve
nix run . -- path/to/title           # run title.exe
nix develop -c bash -c 'make all build/smoke.exe && tests/run-smoke.sh'
```

`nix run . -- path/to/title` is the whole procedure. Each launch:

1. **Stubs what nothing provides.** `rebox-run` scans the exe and the package's DLLs against
   Wine, vkd3d-proton, DXVK and our shims (the report is `$WINEPREFIX/imports.txt`). A missing
   export of one of our shims, or a DLL nobody provides at all, gets a stub that logs its first
   call and returns `E_NOTIMPL`, or 0 (FALSE, NULL) with `ERROR_CALL_NOT_IMPLEMENTED` when mingw's
   Windows headers declare the name returning something other than an HRESULT; the shims are then rebuilt in
   `$WINEPREFIX/shim-build` (only when the stubs change; `REBOX_AUTO_STUBS=0` turns this off).
   An API set (`api-ms-win-*`) that Wine's schema doesn't host, such as an Xbox-only one, would
   stop the exe loading; it gets a DLL of its own that forwards the names Wine's core DLLs export
   and stubs the rest. A name missing from a set Wine does host can't be replaced by a DLL (the
   schema wins), and Wine binds a stub that aborts the process when called; `xmem.dll`, which
   titles import, repoints those imports (in DLLs loaded later too) at a stub that logs the name
   and returns `E_NOTIMPL`
   (`REBOX_IMPORT_FIXUP=0` leaves Wine's).
   Forwarded exports in package DLLs are followed. `nix run . -- --scan path/to/title` prints the report
   without running. To make an entry point real, implement it in `shims/<dll>/` and add it to that
   DLL's `.def`. `tests/run-stubgen.sh` checks the stub generation end to end.
2. **Sets up the prefix:** the shims (including the Gaming Runtime stand-in, which logs to
   `$WINEPREFIX/xgameruntime.log`), `ConsoleMode`, persistent local storage in the prefix, and
   the package as a shadow directory with our FidelityFX stub.
3. **Runs the title.** Console compute and graphics shaders compile through AnyPS5 at their first
   dispatch, draw or mesh dispatch ([docs/anyps5.md](docs/anyps5.md)); pipelines it doesn't take
   (amplification, tessellation) get fallback shaders, so those effects are missing. `D3D12XboxCreateDevice` logs the title's SDK version, which picks the queue layout
   ([docs/d3d12x.md](docs/d3d12x.md)).

The prefix keeps what a run leaves behind: `imports.txt` (the scan), `last-run.log` (everything
the title's process printed: shim logs, crash reports with module+RVA, Wine's errors; the run
before is `last-run.log.1`) and `xgameruntime.log`. Those three files are what to look at, or
send, after a run.

Then iterate: `REBOX_LOG=/path` keeps the shim log, `REBOX_STUB_FATAL=1` stops at the first
unimplemented stub the title reaches, and `xgameruntime.log` lists any Gaming Runtime interface or
method the title asked for that isn't implemented.

### BCPack tables

BCPack, the console's hardware texture compression, is decoded in the title's process by
`shims/xcrdapi/bcpack.c`. Decoding needs a 77,982-byte file of constant tables (entropy models,
index rank tables, BC7 control words and per-format dictionaries). They are Microsoft's format
data, so Rebox doesn't include them: supply your own copy as `bcpack-tables.bin`. The first of
these that exists is used:

1. `REBOX_BCPACK_TABLES`: a path, Unix (`/home/...`) or Windows (`Z:\...`);
2. `bcpack-tables.bin` next to `XCRDAPI.dll` (the Rebox DLLs);
3. `$XDG_CONFIG_HOME/rebox/bcpack-tables.bin`, or `~/.config/rebox/bcpack-tables.bin`.

The log says which file was used, or that none was found. A file of the wrong size, or whose
entropy models don't add up, is reported and not used. Without tables, every BCPack request fails
with `E_NOTIMPL` (zlib and plain reads still work), and textures that need BCPack are missing.

### Environment

| Variable | Meaning |
|---|---|
| `WINE`, `VKD3D_PROTON_DIR`, `DXVK_DIR`, `REBOX_SHIMS_DIR` | Set by the flake wrapper; override to use local builds |
| `WINEPREFIX` | Default `~/.local/share/rebox/<TitleId>`. A prefix made by another Wine build is updated (`wineboot -u`) on the next run; `.rebox-wine` records the build |
| `REBOX_SHADER_CACHE` | Unix path to the cache; default `$WINEPREFIX/shader-cache` |
| `REBOX_DRAW_CENSUS=1` | The per-100-presents log of which console stages draw names draws into every render target (as "other target -1"), not only the display-sized ones: finds a stage, such as a UI text shader, for `REBOX_WATCH_STAGE` or `REBOX_GRAPHICS_DEBUG=magenta:<id>` |
| `REBOX_FRAME_TRACE=<present>[-<present>][,...]` | During those presents (at most 8 ranges), a `trace` line for every console stage compile, in order: present, sequence, thread, kind (compute, vertex, pixel, mesh), code id, stage id, what came of it (ok, past the frame's budget, compiling in the background, past their wait, program failed before, failed, with the reason) and the draw, `ExecuteIndirect`, mesh draw or dispatch that asked, with its counts and its first render target's size and format. Sets a frame's passes beside a reference trace. Every census also counts each kind's outcomes since the last |
| `REBOX_WATCH_STAGE=<id>[:<offset>]` | At that console graphics stage's draws (the first 16, then every 100th), the four floats at the offset (hex, default 0x20) of the buffer its s8 table's first V# names; at its first 4, every buffer V# in that table with its first 8 floats |
| `REBOX_WATCH_TARGET=<address>` | Log when a render target at that title address (hex) is bound, or is a copy's destination (render target views keep what they view through descriptor copies); also counts bound render target handles no view was made at (descriptors a title writes itself) |
| `REBOX_DUMP_STAGES=<id>,...` | With `REBOX_DUMP_SPIRV`, those console graphics stages' code and the first 256 bytes of their s8 table, once each, as `<id>.code.bin` and `<id>.table.bin` |
| (always) | The title's `OutputDebugString` text, its own boot log, is in the log as `title: ...` lines |
| `REBOX_DUMP_SHADERS=1` | Write each shader that misses the cache to `<cache>/dump/<hash>.<stage>.bin` |
| `REBOX_CPU_VISIBLE_BUFFERS=0` | Leave DEFAULT-heap buffers GPU-only (they're then not identity-mapped) |
| `REBOX_ADOPT=0` | Title memory is never made GPU memory in place (buffers placed over it, descriptor heaps in it): to tell problems of memory replaced in place from others |
| `REBOX_CA_BUNDLE=FILE` | CA certificates placed at `G:\etc\ssl\certs\ca-bundle.crt`, where XCurl under WineGDK looks for them (without them every TLS request fails and some hang). The flake passes Nix's `cacert` |
| `REBOX_NATIVE_VCRT=0` | Use Wine's builtin Visual C++ runtime. By default the `msvcp140*`, `vcruntime140*`, `concrt140`, `vccorlib140` and `vcomp140` the package ships next to the exe are loaded (a title crashed at startup in Wine's `msvcp140`) |
| `REBOX_FRAMELESS=0` | Leave the title's top-level windows as it makes them. By default a top-level window with a caption or sizing frame becomes a borderless popup of the size asked for, as the console's screen is (xcl): a title that sizes its back buffers from the client area gets 1912x1046 from a framed window |
| `REBOX_ESCAPE_QUITS=0` | Leave Escape to the title. By default Escape, like closing the title's window, quits: the title is told it is being suspended (when a console title saves) and gets up to `REBOX_QUIT_TIMEOUT` seconds (default 5) before the process ends |
| `REBOX_GAMEINPUT_LOG=1` | Log the controller state the gameinput shim reads (with XInput's raw values), at most 4 times a second while it changes, 200 lines: for a stick that drifts or an axis that reads off-centre |
| `REBOX_EXE=FILE` | Run FILE from the game directory under `MicrosoftGame.Config`'s executable name, e.g. `title.exe.original`, the unpatched exe some dumps keep next to a patched one |
| `REBOX_OVERLAY=DIR` | Lay DIR's files over the package at the same relative paths, in the shadow package only (the dump is untouched), e.g. an Unreal title's `<Project>/Config/UserEngine.ini` |
| `REBOX_ACCELERATION_STRUCTURES=1` | Pass ray tracing acceleration structure builds, postbuild info and copies on to vkd3d-proton. By default they're skipped, since no ray reads them (pipelines are placeholders): a title fills its bottom levels by copies in a mode vkd3d-proton doesn't support, and its top-level build then faulted the GPU |
| `REBOX_DUMP_STATE_OBJECTS=DIR` | Write each distinct serialized ray tracing state object a title loads (`DeserializeStateObjectX`, Xbox device slot 115) to `DIR\<fnv1a64>.bin`. They get a placeholder that traces nothing: console ray tracing code isn't translated yet |
| `REBOX_CONSOLE_MODE` | `ConsoleMode` value `rebox-run` puts in the registry for `XSystemGetDeviceType` (default `0x64`, Series X; `0x50` is Xbox One X) |
| `REBOX_KEYBOARD_PAD=0` | Don't drive the gamepad from the keyboard when no controller is connected (`shims/gameinput`: WASD/arrows sticks, Space A, Left Ctrl B, R X, F Y, Q/E shoulders, Z/C triggers, Enter menu, Tab view, IJKL D-pad) |
| `REBOX_PAD_SCRIPT=<seconds>:<button>,...` | Press gamepad buttons by themselves, for unattended runs: each held for 200 ms that many seconds after `gameinput.dll` loads; buttons `a`, `b`, `x`, `y`, `menu`, `view`, `up`, `down`, `left`, `right` |
| `REBOX_FFX_STUB=0` | Run the package's own FidelityFX loader instead of the stub |
| `REBOX_TEXTURE_ARRAYS=slice-major` | Read placed 2D array textures a slice (its whole mip chain) at a time, addrlib's order, instead of a mip (all its slices) at a time as console texture pools store them |
| `REBOX_CONSOLE_DLLS` | Console system DLLs taken from the dump's system partition (`EmbeddedXvd/Windows/System32`, next to or inside the game directory) instead of being stubbed; default `xg_xs.dll` (the texture-layout library, pure computation). Empty uses none |
| `REBOX_CONSOLE_SYSTEM=DIR` | Where that system partition is, if not found next to the game |
| `REBOX_GAME_ALIAS=0` | Leave out the `GAME:` link in the package directory, which makes console `GAME:\...` paths resolve to the package (the package is also drive `G:`) |
| `REBOX_SHADER_FALLBACK=0` | Fail PSO creation on a missing translation, instead of using fallback shaders |
| `REBOX_STUB_FATAL=1` | Exit on the first unimplemented stub call |
| `REBOX_CONSOLE_COMPUTE=0` | Don't defer console compute shaders to AnyPS5: they're looked up in the shader cache, then get fallback shaders ([docs/anyps5.md](docs/anyps5.md)) |
| `REBOX_CONSOLE_COMPUTE_BISECT=VALUE/BITS` | Run only the console compute programs whose code address hashes to VALUE in its low BITS bits (the rest are skipped), logging each one run: halve the set run by run to find a program that faults the GPU |
| `REBOX_CONSOLE_ASYNC=0` | Compile console shaders in the title's thread at their first dispatch or draw. By default a program's first compile runs on a worker thread while that dispatch is skipped or the draw uses its fallback: large shaders take hundreds of milliseconds each, and a title's game thread can give up waiting for its render thread. Console compute and graphics stages' Vulkan pipelines are made on worker threads too (vkd3d patches 0015 and 0016) |
| `REBOX_CONSOLE_COMPILE_THREADS=N` | Background workers compiling console shaders (default half the CPUs, at most 8); AnyPS5 runs outside the compile lock, so they compile at once |
| `REBOX_PIPELINE_THREADS=N` | vkd3d-proton workers building console pipelines (patch `0015`; default half the CPUs, at most 8). Each build of a world's largest shaders takes gigabytes, so fewer threads bound memory |
| `REBOX_PIPELINE_WORDS_IN_FLIGHT=N` | SPIR-V words of console pipelines vkd3d-proton builds at once (patch `0018`; default 1500000, 0: no limit); a build bigger than that runs alone |
| `REBOX_EAGER_FALLBACK_PIPELINES=1` | Make a Vulkan pipeline for every PSO with fallback shaders at its creation, as before patch `0018` (by default none is made, and their draws without a console pipeline are left out) |
| `REBOX_VARIANT_LOG=1` | Log why console stages get more than one module (d3d12_xs: what differs from the stage's first: AnyPS5's result, host buffers, bindings, SPIR-V) and PSOs more than one console pipeline (vkd3d-proton: which stage or pipeline state differs) |
| `REBOX_INDIRECT_CLAMP=0` | Use `ExecuteIndirect` arguments as they are. By default vkd3d-proton (patch `0017`, [docs/shaders.md](docs/shaders.md)) empties records past `REBOX_INDIRECT_CLAMP_GROUPS` workgroups (default 4194304, or `REBOX_INDIRECT_CLAMP_DIM` a dimension, default 16777216) or `REBOX_INDIRECT_CLAMP_VERTICES` vertices times instances (default 268435456), so an argument buffer that was never filled can't hang the GPU |
| `REBOX_FRAME_COMPILE_BUDGET_MS=150` | How long a frame's compiles of known console shaders may take in the title's recording threads (default 150 ms; 0: no limit; only slow compiles count, see the next). Past it, graphics stages are left out for the rest of the frame, as while a program first compiles, and the next frame goes on from there; compute dispatches always compile |
| `REBOX_FRAME_COMPILE_SLOW_MS=10` | Which compiles count against `REBOX_FRAME_COMPILE_BUDGET_MS`: those that took this long or longer (a new variant, a wait that ran out). Memo hits, 1 to 3 ms each in a world, don't, or a frame's first hundred dispatches spent the budget and left out every later scene draw. 0 counts every call |
| `REBOX_INLINE_COMPILE_WAIT_MS=50` | How long a draw waits for its compile of a known console program (compute dispatches: at least 250 ms; 0: the title's thread compiles it itself, without limit). The compile runs on a pool of its own; past the wait the draw or dispatch is left out and the compile goes on, its next call finding it in AnyPS5's memo. Draws into display-sized targets (UI, loading screens, the composite) always compile on their own thread without limit |
| `REBOX_FIRST_CALLS_WAIT=4` | A console program's first this many calls (draws, mesh draws, ExecuteIndirects, dispatches) wait for their compile instead of being skipped while it compiles, each for up to `REBOX_FIRST_CALL_WAIT_MS`, and their Vulkan pipeline is made at once rather than on a worker (vkd3d patch 0030). A pass the title makes once (the lighting tables a world bakes at its first frame) skipped once stays undone. 0: none wait. Later calls are skipped while compiling, as before |
| `REBOX_SCREEN_PASS_WAIT_CALLS=16` | The same for a full-screen pass's stages (a non-indexed draw of 3 or 4 vertices and one instance), for this many calls, outside the frame limit below |
| `REBOX_FIRST_CALL_WAIT_MS=20000` | How long such a call may wait for its compile (0: no call waits). Every wait of a second or more is logged |
| `REBOX_FIRST_CALLS_FRAME_MS=3000` | How long a frame's first-call waits (other than full-screen passes') may take in all; past it the frame's later first calls are skipped while compiling, as before (a world brings a thousand new programs at once, and a title's watchdog fires when its render thread stops presenting for long). 0: no limit |
| `REBOX_UNKNOWN_BUFFERS=keep` | Bind root views, vertex and index buffers, and read guest buffers (V#s) below 4 GiB or in the GPU memory window as they are when they are in no buffer the title knows (one it made a view of or asked the GPU address of) and no Rebox GPU memory; they are logged either way. By default they read as empty (a root view gets 64 KiB of zeros, a vertex or index buffer none): such an address can only fault the GPU. Known buffers are read no further than their end |
| `REBOX_SPIRV_VALIDATE=1` | Validate every module AnyPS5's output is finalized into (spirv-val); by default the first 256 and then every 32nd are, since validation is much of a compile's time. `=0` validates none |
| `REBOX_ANYPS5_SPECULATE=0` | Turns off AnyPS5's speculation of descriptors and table pointers it can't evaluate (patches/anyps5/0004; on by default, [docs/anyps5.md](docs/anyps5.md)) |
| `REBOX_ANYPS5_NULL_INVALID=0` | Refuses a graphics stage whose T# or V# isn't one when its draw is recorded, instead of binding a null descriptor there and filling the slot at submission (patches/anyps5/0020; on by default, [docs/anyps5.md](docs/anyps5.md)) |
| `ANYPS5_SHADER_CACHE_DIR`, `ANYPS5_NO_SHADER_CACHE` | Where AnyPS5 keeps compiled shaders on disk (default `%LOCALAPPDATA%\rebox\anyps5-cache`), and the compiles that failed, with their errors, so a later run doesn't repeat them (patches/anyps5/0009); or not at all |
| `REBOX_DUMP_SPIRV=DIR` | Writes each console shader's finalized SPIR-V to DIR |
| `REBOX_HEAP_VA=title\|real\|virtual` | What descriptor heaps report as their GPU address (heap slot 10). `title` (default): the shadow heap's 32-bit title address, which AnyPS5 resolves in the console-format copy ([docs/anyps5.md](docs/anyps5.md)). `real`: the real GPU address of vkd3d-proton's descriptors (patched vkd3d-proton with descriptor buffers, as on RADV). `virtual`: vkd3d-proton's own handles. `REBOX_REAL_HEAP_VA=0` also selects `virtual` |
| `REBOX_CRASH_LOG=N` | Report the first N faults (access violations, illegal instructions, divide by zero, stack overflow) with module+RVA, target address, a GPU-window hint, registers and an unwound stack; default 8, 0 turns it off. Reports are first-chance, so a fault the title handles itself also appears |
| `REBOX_SKIP_ASSERTS=1` | Bring-up: step over the title's assertions (`int3` in its own code, as a debugger would) and log each one with its caller, so one run lists every assertion it meets instead of stopping at the first. What follows an assertion is mostly the title's own failure handling |
| `REBOX_SKIP_FAULT=RVA:LENGTH[,...]` | An access violation at that RVA of the exe (hex) skips the faulting instruction (LENGTH bytes, hex) and goes on, logged: for a title's deliberate crash, such as a watchdog writing to address 0, to see what it would do next (bring-up only) |
| `REBOX_HANG_DUMP=N` | When no frame has been presented for N seconds (default 60, 0 off), log every thread's name and unwound stack once, to see what a stuck title waits on; again after the next frame, three times at most |
| `VKD3D_CONFIG=breadcrumbs` | vkd3d-proton's breadcrumbs: when the GPU device is lost, log the commands each queue was running. Before a console dispatch or draw they record the variant's key, which d3d12_xs logs with each compile (`compiled a variant ... key K`), so the dump names the console shader. Slower |
| `REBOX_MESH_PIPELINES=1` | Make mesh pipelines whose console mesh shader is in the amplification slot (a 24 with no 25), renumbered, instead of refusing them (E_INVALIDARG): the target title renders nothing once they exist with fallback shaders (xcl). Their mesh and pixel shaders compile through AnyPS5 at each mesh dispatch ([docs/anyps5.md](docs/anyps5.md)) |
| `REBOX_MESH_USER_DATA_S8=1` | Gives a console mesh stage its root parameters from s8, as before, instead of from s16 (8 system SGPRs, then the user data a vertex stage gets from s0; [docs/anyps5.md](docs/anyps5.md)) |
| `REBOX_NETWORK=1` | Let the title use the network. Off by default: its HTTP connections (WinHTTP, and libHttpClient through it) and sockets fail as with no network, and name lookups other than localhost and the machine's own name fail, so nothing is sent out (a title's crash handler uploads a minidump with the host's details); loopback still works. The Gaming Runtime still reports internet access (`XGR_CONNECTIVITY=none` reports none, which a retail title polls without end). Blocked attempts are logged (`network:`; xmem `import_fixup.c`) |
| `REBOX_WRITE_WATCH=ADDRESS:LENGTH[,...]` | Who writes a range of memory (hex, absolute): once committed it is made read-only, and each instruction that writes into it is logged once (the first 64, with its thread and target; for a range of 64 bytes or less every write, the first 256, with what was there), then let through. A write by the system into it fails, so only for finding a writer (xmem `probe.c`) |
| `REBOX_PHYSICAL_TRACE=<address>` | Log every physical-page mapping or unmapping that covers that hex address, the view remaps around it, and every DirectStorage write into the page mapped there, with thread and caller: what happens to data at an address in physical pages. With `REBOX_WRITE_WATCH` set, the same offset at every other address the page is mapped at is watched too |
| `REBOX_PAGE_CHECK=ADDRESS:LENGTH` | A thread compares that range (hex, up to 1 MiB) with a copy about every millisecond and logs each change (the first 64): bytes changed, the first changed qwords old -> new, and the region type, protection and allocation base then: when and how data at an address changes, writes the watches miss included |
| `REBOX_PROBE=RVA[,RVA...]` | Log calls to the title's own code: an `int3` at each address (hex RVA in the exe, or `MODULE!RVA`), each hit logged with the thread, `rcx`/`rdx`/`r8`/`r9`, `rsp` and the caller, then the instruction single-stepped and the probe rearmed (xmem `probe.c`; up to 64). `REBOX_PROBE_DUMP=N` (up to 64) also logs the first N qwords at `rcx` (`N:rdx`, `N:r8`, `N:r9`, `N:rsi`, `N:rdi`, `N:rbx`, `N:rax`: at that register; `N:rsi+2d80`: from a hex offset), code pointers as module+RVA. `REBOX_PROBE_MEMORY=RVA:BYTES[,...]` also logs those ranges of the exe's globals (hex, up to 0x400 bytes each, as dwords). `REBOX_PROBE_MAX=N` hits are logged per probe (default 20), from hit `REBOX_PROBE_FROM=N` on (default 1) |
| `REBOX_BINK_TRACE=1` | Log what the title asks of the package's Bink 2 DLL (`Bink2*.dll`): each open (name or memory, flags, caller, size, frames, sound tracks, or Bink's error), close, pause, seek, frame buffer registration and the sound and file setup; the first calls of the per-frame ones; and every 5 seconds, per open movie, how often each was called and its state (frame, paused, frame data still being read, read error, sound started). Whether a movie that never shows was opened, is paused, waits on its data or isn't being driven at all |
| `REBOX_STACK_SAMPLE=N[,COUNT]` | Log every thread's stack every N seconds, COUNT times (default 10): where a title that presents, but slowly, spends its time |
| `REBOX_SLOT_STATS=N` | Count calls per D3D12.X proxy slot and log them every N presents and at exit (`c` = our code, `T` = trap, unmarked = forwarded to the PC slot) |
| `REBOX_LOG=/path` | Also append shim logs to this file |
| `REBOX_GPUVA_WINDOW` | `BASE:SIZE` (hex) for identity-mapped GPU memory; default `200000000000:10000000000`; `off` stops xmem reserving it |
| `REBOX_GPUVA_TRACE=FILE` | The identity layer appends a line per allocation (`A`) and free (`F`; `f` for title memory made GPU memory) in its window: wall time, address, size, memory type. Traces a GPU fault address in the window to what owned it |
| `REBOX_FOSSILIZE_DB=DIR` | Record every pipeline the run creates into `DIR` with Valve's Fossilize layer (written as they're made, so a crash keeps them). With `REBOX_FOSSILIZE_REPLAY=N`, `rebox-run` first merges earlier recordings into `DIR/merged.foz` and builds them again on `N` threads with `fossilize-replay`, filling the driver's disk cache (set `MESA_SHADER_CACHE_DIR` and a large `MESA_SHADER_CACHE_MAX_SIZE`), so the run finds those pipelines there instead of compiling them. `REBOX_FOSSILIZE_REPLAY_S` bounds the replay (default 900 s) |
| `REBOX_GPU_MEMORY=0` | Give GPU-visible `XMemAlloc` types plain `VirtualAlloc` memory instead of D3D12 heaps |
| `REBOX_DIRTY_PAGES=0` | Don't reserve title memory (`XMemVirtualAlloc`, `XMemAlloc` pages) with `MEM_WRITE_WATCH`: placed textures there are found changed by fingerprints, as in GPU memory, instead of by the pages written before each submission ([docs/d3d12x.md](docs/d3d12x.md) "Texture changes") |
| `REBOX_TEXTURE_TRACKING=0` | Don't write-track placed textures in physical-page views (their pages read-only, the first write to each 64 KiB page noted): they are found changed by fingerprints again, the hot ones every submission ([docs/d3d12x.md](docs/d3d12x.md) "Texture changes") |
| `REBOX_VIEW_RETIRE=0` | Keep every texture made for a view of memory (slots 70, 71), as before: by default one with memory of its own is released, once the GPU is past it and no descriptor slot holds a view of it, when a view texture of another shape is made over its memory (a streaming world reusing its texture pools grew them without end) |
| `REBOX_STOCK_VULKAN=1` | Don't switch AMD GPUs to `.#radv-rebox` |
| `REBOX_PRESENT=0` | No window or swapchain for `PresentX` (frame dumps still work) |
| `REBOX_VSYNC=N` | `PresentX` sync interval, default 1. Frames the display isn't taking (a hidden Wayland window) are dropped, never waited for |
| `REBOX_PRESENT_SCALE=0` | Present the plane at its own size. By default a plane that isn't the window's size (4K in a 1920x1080 tile) is scaled to it with a blit, since vkd3d-proton's swapchain showed it unscaled (a quarter of it); the swapchain is the window's size, frame dumps stay the plane's |
| `REBOX_PRESENT_FIT=1` | Make a presentation window larger than its monitor's work area (titles size theirs from the 4K plane) borderless over the monitor, which Wine shows fullscreen. `REBOX_PRESENT_FIT=WxH` gives it that size instead. Off by default: the window stays as the title made it, a tiling compositor sizes it anyway, and the plane is stretched into it. The log gives the window and monitor sizes and the DPIs Wine reports |
| `REBOX_PRESENT_WAIT=1` | Let vkd3d-proton wait for presents to reach the screen (`VK_KHR_present_wait`/`present_wait2`). Off by default: with a window the compositor isn't showing (Wayland) those waits never end and vkd3d-proton stops the D3D12 queue behind them at the next swapchain rebuild |
| `REBOX_CONSOLE_GRAPHICS=0` | Don't defer console vertex and pixel shaders to draw time (AnyPS5, vkd3d-proton patch 0006); their PSOs keep the fallback shaders, which draw nothing |
| `REBOX_GRAPHICS_DEBUG=magenta` | Every deferred console pixel shader exports magenta instead: shows whether draws reach the render targets at all, apart from whether the real pixel shaders' output is right; `magenta:<id>,<id>` replaces only the pixel stages with those ids (as the log names them), `magenta-except:<id>,...` every pixel stage but those; `none:<id>,...` makes those stages write nothing (what a full-screen pass such as a fade covers) |
| `REBOX_DEBUG_PS_CUT=<id>:<dword>:<r>,<g>,<b>` | That console pixel stage's own code up to the dword offset, then an export of VGPRs r, g, b (alpha 1): halving the offset run by run finds an instruction translated wrongly |
| `REBOX_DEBUG_PS_LZ=<id>,...` | Those console pixel stages run their own code with every `image_sample` made `image_sample_lz` (level 0, no derivatives): whether implicit-LOD sampling is what fails; `REBOX_DEBUG_PS_NOWQM=<id>,...` makes their whole-quad-mode prologue (`s_mov_b64 s[N:N+1], exec`, `s_wqm_b64`, `s_and_b64 exec`) `s_nop` instead |
| `REBOX_DEBUG_VS_WHITE=<id>,...` | Those console vertex stages run their own code but export param0 (a vertex colour they pass on) as 1: whether that colour is what blackens a draw; `REBOX_DEBUG_VS_UV=<id>,...` exports param1 as 0.5 instead (texture coordinates at the centre) |
| `REBOX_DEBUG_VS_CONST=<id>,...` | Those console vertex stages draw a full-screen triangle exporting param0 = (1, 0, 1, 1) and param1 = 0.5: with `REBOX_GRAPHICS_DEBUG=attr<N>` on the pixel stage, whether parameters reach it |
| `REBOX_GRAPHICS_DEBUG=attr<N>:<id>,...` | Those console pixel stages export their interpolant N (a vertex colour, texture coordinates) as RGB instead; `texture<hex>:<id>,...` exports as grey the first channel of the texture whose T# is at that offset of their s8 table, sampled at interpolant 1 (`texturec<hex>`: at its centre): whether the stage gets its colours and its textures hold anything |
| `REBOX_GRAPHICS_DEBUG=fullscreen` | Magenta pixel shaders, and every deferred console vertex shader replaced by a full-screen triangle from the vertex ID: shows whether draws reach the presented image at all, apart from the real vertex shaders' positions |
| `REBOX_FRAME_PACING=0` | Make `SetFrameIntervalX`/`ScheduleFrameEventX`/`WaitFrameEventX` no-ops, so the title runs as fast as `PresentX` allows |
| `REBOX_DUMP_FRAMES=DIR` | Write presented frames to `DIR/frame_<n>.bmp` (8-bit RGBA/BGRA planes) or `.bin` |
| `REBOX_DUMP_FRAMES_AT` | Which frames to dump, comma-separated; default `1,60,600` |
| `REBOX_DUMP_FRAMES_EVERY=N` | Also dump a frame every N seconds (the first frame, then the first one N seconds after the last dump) |
| `REBOX_DUMP_TITLE_TEXTURE=<address>` | With `REBOX_DUMP_FRAMES`, write the title memory of the placed texture at that hex address as `DIR/texture_<address>_<n>.bin` each time the texture state log runs (every 600 submissions, eight times) |
| `REBOX_DUMP_GPU_TEXTURE=<address>` | With `REBOX_DUMP_FRAMES`, at each dumped frame also read back the GPU contents of the texture placed at that hex title address, as `DIR/texture_<n>.bmp`, and log its mean value: whether a target the title samples was drawn at all |
| `REBOX_BCPACK_TABLES=PATH` | The BCPack table file ([BCPack tables](#bcpack-tables)) |
| `REBOX_BCPACK_DUMP=DIR` | Write each BCPack chunk a request decodes (after zlib, when a request has both) to `DIR/bc<codec>-<n>-<destination bytes>.bin`, at most `REBOX_BCPACK_DUMP_MAX` (default 4096) of them: real packed data for `build/bcpack_test decode` |
| `REBOX_DSTORAGE_IGNORE_FAILURES=1` | A DirectStorage request that fails (logged and counted as before) no longer makes the next status entry report the failure: whether a title gives up on a whole batch for one failed request (a BCPack block the decoder refuses) |
| `REBOX_DISPATCHX=1` | Console DispatchX calls (list slot 80, 78 on older lists) run as Dispatch; by default they are dropped, since the title's ray tracing builder dispatches this way and run, they faulted the GPU |
| `REBOX_MEMORY_WAITS=0` | Console memory waits in lists (Wait32/64BitValueX) are ignored; lists are ordered only by the title's fences: whether the waits are what stalls a title's queues |
| `REBOX_MEMORY_WAIT_TIMEOUT_MS=<ms>` | How long a queue stands at a GPU memory wait (a counter wait made a fence wait) before its value is signalled from the CPU and the wait logged; default 2000. The time runs from when the queue gets to the wait, not from the submission |
| `REBOX_MEMORY_WAIT_FORCE=0` | A GPU memory wait the queue stands at, whose value no queue has been told to signal, is no longer released at the next present |

The shader cache ([docs/shaders.md](docs/shaders.md)) is keyed by the 64-bit FNV-1a hash of the
bytecode exactly as the title passes it to `Create*PipelineState`.

Console shaders are DXBC containers too, so the shim doesn't treat every `DXBC` blob as PC
bytecode. Only a container with a `SHEX`/`SHDR` part or a `DXIL` part holding LLVM bitcode passes
through. Everything else, including the console's `DXIL` parts (a `NOBC` stub around zlib `FSSM`
GFX10.3 code), is looked up in the cache.

A miss would fail PSO creation, which a title rarely survives. So by default the whole pipeline
gets fallback shaders instead (`generated/fallback_shaders.h`, from `tools/gen_fallback_shaders.sh`):
graphics draws nothing, compute does nothing, and the effect is just missing. Each one is logged.
Mesh pipelines get a mesh shader that emits nothing and an amplification shader that launches
nothing (DXIL from `dxc`). `REBOX_SHADER_FALLBACK=0` makes a miss fail instead, for hunting
down missing translations.

### GPU faults

A GPU fault ends a run with `radv: GPUVM fault detected at address ...` and the client (`TCP` for
vector memory, `SQC (data)` for scalar loads), but not what owned the address or what was running.
`.#radv-rebox` (`patches/mesa-radv-fault-report.patch`) adds a report for that, cheap enough for
the world to run at near-normal speed:

```sh
RADV_DEBUG=faultreport
RADV_DUMP_DIR=$HOME/code/gta/runs-target/radv-dumps
```

| Variable | Meaning |
|---|---|
| `RADV_DEBUG=faultreport` | Keeps RADV's BO history (every BO made and freed, with its VA range and a timestamp) and the address binding reports (which `VkBuffer`, `VkImage` or memory had each range) in memory, and records the pipeline each queue type last bound (one `WRITE_DATA` per pipeline bind). When a submission finds the device lost, `radv_dumps_<pid>_<time>/` is written once: `fault.log` (the GPUVM fault and its status; every BO history entry covering the address, and the live BOs nearest below and above it, so a read past a BO's end or into a freed one shows as such; the address binding reports covering it; the GFX and COMPUTE rings' last bound pipelines: `VkPipeline` handle, hash, each stage's code range, user SGPR layout and code, `pipeline_<handle>_<stage>.bin`), `bo_history.log`, `addr_binding_report.log`, `app_info.log`, `gpu_info.log` and `dmesg.log`. Timestamps are `CLOCK_MONOTONIC` nanoseconds; `fault.log` gives monotonic and wall time at the report to line them up with the log. The command processor runs ahead of the shaders, so the last bound pipeline is the faulting one or one bound shortly after it (the last barrier before the fault bounds it) |
| `RADV_DUMP_DIR=DIR` | Where RADV's reports go (made if missing): `faultreport`'s, `hang`'s and the trap handler's `radv_dumps_*` directories, and `RADV_DEBUG=bo_history`'s live log (`radv_bo_history_<pid>.log`, else `/tmp/radv_bo_history.log`). Without it they go to `$HOME`, which the run loop must not change |
| `REBOX_PIPELINE_HANDLES=1` | vkd3d-proton (patch `0031`) logs each console pipeline's `VkPipeline` handle with its compiles' keys (`rebox: console graphics pipeline 0x...: keys VS PS`, `mesh`, or `compute` with the variant's two keys), which d3d12_xs logs its compiles under (`compiled a variant ... key K`): the handle in `fault.log` names the console shaders. On by itself with `RADV_DEBUG=faultreport` |

What the other modes cost, for comparison: `RADV_DEBUG=hang` waits for the GPU to go idle after
every draw and dispatch (it turns on `syncshaders`) and after every submission, and submits one
command buffer at a time, so a world crawls and the title's watchdog fires first (it also writes to
`$HOME`, now `RADV_DUMP_DIR`). `RADV_DEBUG=bo_history` alone writes and flushes a line per BO made
or freed, with no report at the fault. `VKD3D_CONFIG=breadcrumbs` slows placed resource creation
to a crawl. `VK_EXT_device_fault` (vkd3d-proton queries it at device loss) gives only the faulting
address, and its vendor binary only with `hang`. `faultreport` costs a few list appends per BO, one
packet per pipeline bind and memory for the history (tens of bytes a BO, a little more a buffer).

A fault report's address and BO history read together with the identity layer's trace
(`REBOX_GPUVA_TRACE`) for the GPU memory window, and with the log's `compiled a variant` lines
for the pipeline's console shaders.

## What the tests cover

- Shims build with mingw. `tests/run-smoke.sh` passes under stock Wine 11.21 + vkd3d-proton 3.0.1
  on lavapipe. Bytes vkd3d-proton can't compile succeed in both `CreateComputePipelineState` and
  `CreatePipelineState` (stream) when a cache entry exists, and, with fallbacks off, fail with
  `E_FAIL` when it doesn't.
  The same holds for a synthetic console container (`DXIL` part with a `NOBC`/`FSSM` stub). Real
  PC DXBC compiles directly even when a broken cache entry exists for its hash.
- `build/bcpack_test` (run first by `tests/run-smoke.sh`, natively under ASan and UBSan) tests
  the BCPack decoder with synthetic tables. It decodes the format notes' two reference vectors byte
  for byte. A synthetic encoder builds streams that round trip through the decoder and through a
  separate model of the lane schedule: multi-fragment lanes with and without reset, alphabet
  reductions, model shapes, long lengths, all 16 lanes competing for the 8-byte budget. Every format
  and every BC6H and BC7 mode, rotation and partition is checked field by field with standard block
  readers, including references across the 256-block history and into the dictionaries. Every error
  condition is checked too: truncation at every length, trailing bytes (ignored), and noise.
  `REBOX_BCPACK_TABLES` adds checks on a real table file. The smoke test decodes BC4 through
  `XCRDAPI.dll`, from a file offset and after zlib.
- Every import of the console DLLs titles bundle resolves under Wine + vkd3d-proton + shims.
  `XMemGetWorkingSetStatistics` answers the selector the AMD upscaler passes (`0x20000001`):
  TRUE, with private commit in the u64 at `+8` (the call shape is from Xbox2Windows's call-site
  notes).
- `XMemVirtualQuery` takes `(process, address, info)` with a 0x40-byte info struct, as
  XboxBvhBuilder.dll calls it, and the smoke test covers that call.
- The FidelityFX stub answers a render-resolution query as FSR does. It scales a cleared 32x24
  render into a 64x48 output through `ffxDispatch` on an Xbox command list, and the presented
  frame has the render's colour.
- Fallback shaders: a graphics PSO built from untranslated console bytes is created, a draw with
  it leaves the frame untouched, and compute and stream PSOs work too. Strict mode still rejects
  them.
- `PresentX` presents a cleared render target recorded through Xbox slots. Both plane-param
  layouts work (resource, or pointer to it), and the dumped frame has the cleared pixels. This
  was checked headless; the window and swapchain need a display.
- The smoke test drives the Xbox proxies by slot number, as title code does. It creates the
  device, queue, list and heap, records through the shifted command-list slots, executes and
  waits on a fence, and checks the zeroed inline fields, the RAX-returning heap starts and the
  trap on an unknown slot. It passes on lavapipe, also with the identity layer loaded (the layer
  passes through there).
- On a Radeon 890M with `.#radv-rebox` and the identity layer (`tests/manual-test.sh`), the
  Wine smoke test passes and every GPU heap `d3d12_xs` hands out is identity-mapped, including
  heaps allocated right after a free.

## Known gaps (ordered roughly by how soon they'll bite)

- **Console shaders are partly covered** ([docs/anyps5.md](docs/anyps5.md)). Compute, vertex,
  pixel and mesh shaders compile through AnyPS5 at their first dispatch, draw or mesh dispatch
  (mesh ones haven't drawn on a GPU yet, and need `REBOX_MESH_PIPELINES=1`). Amplification,
  tessellation and geometry pipelines get fallback shaders, shaders that need LDS spilling or
  scratch are skipped, and so are shaders using instructions AnyPS5 doesn't handle yet; each is
  logged once.
- **D3D12.X interfaces** ([docs/d3d12x.md](docs/d3d12x.md)). `D3D12XboxCreateDevice` returns
  proxies with the Xbox layouts. Their slot tables are merged from our disassembly and
  [ScarlettX](https://github.com/Freminet/ScarlettX)'s traces. Slots nobody has identified
  log and return `E_NOTIMPL`; `REBOX_STUB_FATAL=1` stops at the first one. `PresentX` shows
  plane 0 in a window through a DXGI swapchain. Not done yet:
  - only plane 0 is shown. In Microsoft's public samples a second plane is an SDR copy for
    GameDVR next to an HDR plane 0, and the runtime stand-in reports HDR off, so a title should
    present one SDR plane; if it presents two, the log says so, and plane 0 may be HDR10 shown
    as SDR. Planes in formats a swapchain can't
    take, such as R11G11B10, are converted to RGBA8 with a blit: HDR is clamped, not tone-mapped;
  - frame events come from a CPU clock, not the display: `WaitFrameEventX` follows the period the
    title sets, but isn't locked to the monitor's vblank, so with vsync on a mismatched refresh
    rate can judder;
  - the console's rectangle lists (topology 17, used for full-screen passes) are drawn by AnyPS5's
    rect-list stages, a tessellation pair that makes each rectangle's fourth corner after the
    vertex shader (vkd3d-proton patch 0011); a draw whose pipeline isn't deferred console graphics
    covers only its first triangle, half of each rectangle. `REBOX_RECT_LISTS=cpu` makes the
    fourth corner on the CPU from the vertex buffers instead (after xcl);
- **Unified memory** ([docs/d3d12x.md](docs/d3d12x.md)). Titles use GPU VAs as CPU pointers and
  the reverse. GPU-visible `XMemAlloc` memory and command-list embedded data come from
  CPU-visible D3D12 heaps. `rebox-run` loads `VK_LAYER_REBOX_gpuva_identity` and, on AMD,
  `.#radv-rebox`, which together make those heaps identity-mapped. Without them,
  `d3d12_xs` logs that GPU memory is not identity-mapped. Buffers the title creates in DEFAULT
  heaps, and DEFAULT heaps that hold only buffers, are made CPU-visible (custom `WRITE_BACK`/`L0`),
  so they're identity-mapped too (`REBOX_CPU_VISIBLE_BUFFERS=0` opts out); the title still sees
  DEFAULT from `GetHeapProperties` and a heap's `GetDesc` (Unreal 5 takes a CUSTOM heap's path that is
  broken on the console). Textures and mixed
  heaps keep ordinary GPU addresses. The title's own memory becomes GPU memory where it is when a
  buffer is placed over it or a descriptor heap lives in it (spans of 2 MiB to 256 MiB, through the
  layer's mailbox); memory it never places anything over stays CPU-only, so a shader reading it
  through a pointer the title built itself reads nothing. Physical pages are shared sections, so a
  page mapped at two addresses is the same memory at both, unless one address lies outside the
  ranges reserved for physical pages (there it gets a copy).
- **Ray tracing and depth UAVs are placeholders.** Console ray tracing pipelines
  (`DeserializeStateObjectX`, `CreateStateObject`) get placeholder objects that trace nothing,
  and their `DispatchRays` are skipped, as are state objects the title sets that aren't ours
  (`REBOX_DUMP_STATE_OBJECTS` saves the blobs to work out their format). Acceleration structure
  builds and copies are skipped too, since no ray reads them; inline ray tracing in console shaders and the console BVH builder's output aren't
  translated either. A depth texture that's also a UAV (Unreal 5's Nanite) is a depth target only,
  so writes through its UAV views are lost; TuxBox pairs it with an R32 image and copies the
  depth plane across, which is the fix. Both matter for titles that use ray tracing or Nanite.
- **Controllers SDL maps wrongly.** Wine reads controllers through SDL, by SDL's database of
  layouts. For an Xbox One S controller on Bluetooth (045e:02e0, version 0408) that entry has
  another axis order than the kernel's (the right stick's Y reads the right trigger, at full
  deflection at rest); `rebox-run` sets a corrected
  `SDL_GAMECONTROLLERCONFIG` for it, with its buttons numbered as the kernel reports them.
  Another controller read wrongly needs its own line in `SDL_GAMECONTROLLERCONFIG`;
  `REBOX_GAMEINPUT_LOG=1` shows what the title is given, with XInput's raw values.
- **Audio: XMA2 decoding is untested on real data.** Under WineGDK, FAudio sends XMA2 to Wine's
  WMA decoder, which accepts only WMA formats, so creating an XMA2 source voice fails.
  `shims/xaudio2_9` wraps Wine's `xaudio2_8` (same FAudio code) and turns XMA2 voices into PCM
  voices. Each XMA2 buffer is decoded at submission by FFmpeg's `xma2` decoder: the flake's
  `ffmpeg-xma2`, a static mingw libavcodec with only that decoder and Opus's. If decoding fails, or the DLL is
  built without `FFMPEG_XMA2`, the voice plays silence of about the buffer's length. A callback
  between the voice and the title keeps callback contexts intact and frees the PCM.
  FFmpeg has no XMA2 encoder, so no test feeds it real XMA2. Without an audio output device the
  mastering voice fails as on Windows (Wine would report success and FAudio crash later); Unreal
  titles treat that as fatal, so a machine without a sound card needs a sound server, e.g. a
  PulseAudio null sink. Bink's audio is decoded to PCM by Bink and is
  unaffected. Wine also reports a mastering voice as created when there's no audio device, after
  which FAudio crashes on the first source voice; that only matters on machines without audio.
- **Audio: the console's decoder queue (`xapu.dll`) decodes Opus.** FMOD in a GDK title sends it
  Opus packets per stream and takes back 48 kHz float PCM (VIctory, GPL-3.0, facts only). Each
  Process command is decoded when it is queued by the same FFmpeg build's Opus decoder; a packet it
  refuses, or a build without `FFMPEG_XMA2`, gives silence of the length asked for. The audio DSP
  (`chad.dll`) and the coprocessor's SHAPE contexts (`acphal.dll`) still do nothing, so whatever a
  title runs through them is lost.
- **Upscaling is bilinear.** The console FidelityFX upscaler creates its passes from the console
  driver's own container (FSPX), which `d3d12_xs` defers to AnyPS5 like a title's shaders
  ([docs/anyps5.md](docs/anyps5.md)); whether its passes compile and run hasn't been tried, so
  the stub is the default. `rebox-run` runs the title from
  `$WINEPREFIX/package`, a directory of symlinks to the package, with our
  `amd_fidelityfx_loader_xbox.dll` in place of the package's. That stub answers FSR's queries
  (quality-mode ratios, but no jitter) and scales the colour input into the output with a
  compute blit. There's no temporal anti-aliasing; frame generation reports no provider.
- **BCPack needs tables from you, and isn't checked against the console.** BCPack, the console's
  hardware texture compression (about 14% of one title's catalogued textures), is decoded in
  process by `shims/xcrdapi/bcpack.c` (BC1, BC3, BC4, BC5, BC6H, BC7), written from a description
  of the format. Its tables are Microsoft's data and not in the repository
  ([BCPack tables](#bcpack-tables)); without them BCPack requests fail. On real chunks from one
  title's texture pools, BC1 matches an independent decoder (Leonida's, run as a separate program)
  byte for byte: 100 chunks, 2.1 million blocks. That decoder lacks the dictionary entries the BC7
  chunks reference, so BC7 was only checked by eye and by how smooth its blocks come out. BC3,
  BC4, BC5 and BC6H have no real-data check yet. Decoding runs at about 40 MB/s of blocks per
  core. Swizzled destinations and physical-page sources in DirectStorage requests aren't
  supported.
- **Textures placed at title addresses are copies.** On the console a texture placed with
  `CreatePlacedResourceX` is the memory at its address, in a Gfx10 tiled layout. Here it has its
  own memory, filled from the title's: `d3d12_xs` detiles it with AMD's addrlib (Mesa, MIT) when
  it is created and whenever DirectStorage reports writing there, before the next submission.
  In title memory xmem reserves with write watching (`XMemVirtualAlloc`, `XMemAlloc` pages), the
  pages written since the last submission make their textures upload, as xcl does
  ([docs/d3d12x.md](docs/d3d12x.md) "Texture changes"). In physical-page views xmem keeps the
  textures' pages read-only and notes the first write to each (and each page mapped anew), and
  those textures upload; each is fingerprinted at most every 4 seconds as a safety net
  (`REBOX_TEXTURE_TRACKING=0` turns tracking off). Elsewhere (GPU memory, title memory made GPU
  memory in place) writes nothing reports (the CPU, GPU copies to the address) are caught by sampling: each
  submission re-checks a fingerprint of 32 points in up to 512 textures (round robin), which sees
  a texture rewritten but not a small edit between the points (`REBOX_TEXTURE_WATCH=0` turns
  it off). A texture placed on memory that isn't committed yet waits until it is (rechecked after
  1, 2, 4, ... up to 1024 submissions). `REBOX_TEXTURE_SWIZZLE=auto` (default): a console
  `Layout` from 0x100 is 0x100 plus the swizzle mode (0x100 linear, 0x105 `SW_4KB_S`, 0x10A
  `SW_64KB_D`); Layout 2 or desc flag 0x08000000 `SW_64KB_D`; otherwise cubemaps, volumes and
  plain 2D textures `SW_4KB_S` and other arrays `SW_64KB_D` (xcl, TuxBox, measured texture pools); each
  distinct layout is logged; `64KB_D`, `4KB_S`, `linear` or `off` force one. A texture a GPU copy
  writes (an Unreal 5 title's font atlas) stops being uploaded from title memory, and textures up to 4 MiB
  are compared whole, so small edits are seen. Textures the CPU has rewritten since they were made (a
  glyph atlas, a video's planes) are checked every submission: whole up to 32 MiB of them, larger ones sampled. Render target, depth and UAV textures are the GPU's and
  aren't filled, except small single-sample colour targets (64x64 to 1024x1024): a title's glyph
  caches are such targets the CPU fills (`REBOX_SYNC_SMALL_TARGETS=0` treats them as the GPU's).
- **Console root signatures and pipeline subobjects** pass through unchanged. Console root
  signatures set flags `0x03000000`. Windows' runtime rejects those bits, but vkd3d-proton
  ignores them, and the smoke test checks that it still does. Their meaning is unknown. Pipeline
  streams are parsed up to the public type 28 (DEPTH_STENCIL2, RASTERIZER1/2 included). A stream
  with a type beyond that, such as an Xbox-specific one, skips substitution and is logged once,
  because its size is unknown.
- **XALLOC_ATTRIBUTES** decoding follows [SlimEra](https://github.com/XWine1/SlimEra)'s Xbox One
  ERA definition (alignment 2^n, at least 16 bytes), which matches the 32-byte alignment Bink relies
  on. Allocations with 64 KiB or 4 MiB pages start on that boundary. Scarlett's header hasn't been
  seen, so the meaning of graphics memory types beyond "GPU-visible" is unconfirmed. `XMemAlloc` memory is always zeroed, including blocks
  reused from freed ones.
- **WineGDK was archived 2026-11-13.** Its successors are
  [xodus-gaming/wine](https://github.com/xodus-gaming/wine) and
  [xodus-gaming/xgameruntime](https://github.com/xodus-gaming/xgameruntime), but that runtime only
  covers threading so far. Switch the `winegdk-src` input once it catches up.

## Licence

Rebox is licensed under the GNU General Public License, version 2 only
([LICENSE](LICENSE)), so it can load AnyPS5's shader recompiler (GPL-2.0-only) in-process.
BCPack's method (its interleaved rANS entropy layer) is covered by Microsoft's patent
US 11,234,023 B2, whatever the code's licence; its tables are Microsoft's data and are never part
of the repository.
The patches in `patches/` modify vkd3d-proton (LGPL-2.1+), Mesa (MIT), Wine (LGPL-2.1+) and AnyPS5
(GPL-2.0-only) and are offered under
those projects' licences too. Game data, dumps and anything derived from them (`build/`) are never
part of the repository.
