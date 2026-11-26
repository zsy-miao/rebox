#!/usr/bin/env bash
# Runs build/smoke.exe under Wine against the built shims and vkd3d-proton.
# Needs: WINE, VKD3D_PROTON_DIR, DXVK_DIR, REBOX_SHIMS_DIR, vkd3d-compiler on PATH.
# Without a GPU, point VK_ICD_FILENAMES at lavapipe (the flake's devShell does).
set -euo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
# A dev shell exports $out (its output path); the log captured in $out below must not be passed to
# every command's environment (over 128 KiB, exec fails with E2BIG).
unset out
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

: "${WINE:?}" "${VKD3D_PROTON_DIR:?}" "${DXVK_DIR:?}" "${REBOX_SHIMS_DIR:?}"

# The BCPack decoder's unit tests (native, synthetic tables). REBOX_BCPACK_TABLES naming a table
# file (a Unix path) adds checks on it.
echo "== BCPack decoder"
bcpack_tables=()
[[ -f ${REBOX_BCPACK_TABLES:-} ]] && bcpack_tables=("$REBOX_BCPACK_TABLES")
"$here/build/bcpack_test" "${bcpack_tables[@]}" | grep -v '^     '

# "Console" shader: bytes no D3D12 implementation will accept.
printf 'REBOX-FAKE-CONSOLE-SHADER-%s' "$(date +%s)" > "$work/fake.bin"

# Its "AnyPS5 translation": a real cs_5_0 DXBC blob.
printf '[numthreads(1, 1, 1)] void main() {}\n' > "$work/cs.hlsl"
vkd3d-compiler -x hlsl -b dxbc-tpf -p cs_5_0 -o "$work/translated.dxbc" "$work/cs.hlsl"

# Console shaders arrive as DXBC containers too: a DXIL part holding a NOBC stub around zlib FSSM
# (GFX10.3 code). They must be looked up in the cache, not passed through as PC bytecode.
python3 - "$work/console.dxbc" <<'PY'
import struct, sys, zlib
fssm = b"FSSM" + bytes(0x80) + struct.pack("<2I", 0xBFA00003, 0xBF810000)
part = b"XBCH" + bytes(0x60) + b"NOBC" + bytes(0x14) + zlib.compress(fssm)
part += bytes(-len(part) % 4)
body = b"DXIL" + struct.pack("<I", len(part)) + part
header = b"DXBC" + bytes(16) + struct.pack("<4I", 1, 0x24 + len(body), 1, 0x24)
open(sys.argv[1], "wb").write(header + body)
PY

mkdir -p "$work/cache"
# The cache key: FNV-1a 64 of the bytecode (shims/d3d12_xs/shader_cache.h).
hash() {
    python3 -c 'import sys
h = 0xcbf29ce484222325
for b in open(sys.argv[1], "rb").read():
    h = ((h ^ b) * 0x100000001b3) & 0xffffffffffffffff
print(f"{h:016x}")' "$1"
}
for f in fake.bin console.dxbc; do
    cp "$work/translated.dxbc" "$work/cache/$(hash "$work/$f").dxil"
done
# PC bytecode must never be substituted: give it an entry that would break pipeline creation.
printf 'not a shader' > "$work/cache/$(hash "$work/translated.dxbc").dxil"

export WINEPREFIX="$work/prefix" WINEARCH=win64 WINEDEBUG=${WINEDEBUG:--all}
"$WINE" wineboot -i >/dev/null 2>&1
sys32="$WINEPREFIX/drive_c/windows/system32"
install -m 644 "$VKD3D_PROTON_DIR"/d3d12.dll "$VKD3D_PROTON_DIR"/d3d12core.dll "$DXVK_DIR"/dxgi.dll "$sys32"/
install -m 644 "$REBOX_SHIMS_DIR"/xmem.dll "$REBOX_SHIMS_DIR"/d3d12_xs.dll "$REBOX_SHIMS_DIR"/xaudio2_9.dll \
    "$REBOX_SHIMS_DIR"/amd_fidelityfx_loader_xbox.dll "$REBOX_SHIMS_DIR"/pixevt.dll \
    "$REBOX_SHIMS_DIR"/xcrdapi.dll "$REBOX_SHIMS_DIR"/gameinput.dll "$sys32"/
export WINEDLLOVERRIDES="d3d12,d3d12core,dxgi,xaudio2_9,gameinput=n"
export REBOX_PAD_SCRIPT=0:x  # gameinput: X held just after the DLL loads (smoke.c checks it)
# The hand-written console compute programs here read their table at s[0:1], the earlier layout
# (vkd3d-proton patch 0007 defaults to the console's, measured on a UE5 sample).
export REBOX_CONSOLE_COMPUTE_LAYOUT=pairs
# Its checks read a dispatch's results at once: compiles in the calling thread (one test turns
# background compiles on itself).
export REBOX_CONSOLE_ASYNC=0
# View textures are evicted only when smoke.c asks (test_view_texture_evict), so other checks of
# view textures alive aren't raced by the periodic pass.
export REBOX_VIEW_IDLE_S=0 REBOX_VIEW_TEXTURE_BUDGET_MB=0
# View buffers nothing writes share a zeroed heap (opt-in), so test_view_buffer_retire checks it.
export REBOX_VIEW_BUFFER_SHARE=1
# AnyPS5's recompiler (docs/anyps5.md): compiles a built-in compute shader.
if [ -n "${REBOX_ANYPS5_DIR:-}" ]; then
    install -m 644 "$REBOX_ANYPS5_DIR"/rebox_anyps5.dll "$sys32"/
    "$WINE" "$REBOX_ANYPS5_DIR"/sa5_check.exe | grep -q "^ok   compiled" \
        || { echo "FAIL rebox_anyps5.dll did not compile the built-in shader"; exit 1; }
    echo "ok   rebox_anyps5.dll compiles a compute shader under Wine"
fi
export REBOX_SHADER_CACHE="Z:${work//\//\\}\\cache"
# PresentX frame dumps, checked by the smoke test.
mkdir -p "$work/frames"
export REBOX_DUMP_FRAMES="Z:${work//\//\\}\\frames" REBOX_DUMP_FRAMES_AT=1,2,3,4

# SMOKE_ONLY=<file> runs just that case (for debugging).
for f in fake.bin console.dxbc translated.dxbc; do
    [[ -n ${SMOKE_ONLY:-} && $SMOKE_ONLY != "$f" ]] && continue
    echo "== $f"
    "$WINE" "$here/build/smoke.exe" "Z:${work//\//\\}\\$f"
done
[[ -n ${SMOKE_ONLY:-} ]] && exit 0

# The GPU VA layer's memory report (layers/gpuva_identity.c), every second here: the Linux side's
# resident memory, glibc's heap, device memory and the pipelines made through it.
if [[ -f $here/build/librebox_gpuva_identity.so ]]; then
    echo "== translated.dxbc, GPU VA layer's memory report"
    out=$(VK_ADD_LAYER_PATH="$here/build" VK_INSTANCE_LAYERS=VK_LAYER_REBOX_gpuva_identity REBOX_GPUVA_MEMORY_S=1 \
          "$WINE" "$here/build/smoke.exe" "Z:${work//\//\\}\\translated.dxbc" 2>&1) || { echo "$out" | tail -40; exit 1; }
    grep -E "^\[rebox gpuva\] memory: resident [1-9][0-9]* MiB .*glibc malloc [0-9]+ MiB in use.*pipelines [0-9]+ alive \([1-9][0-9]* made" <<<"$out" | tail -1
    grep -Eq "^\[rebox gpuva\] memory: resident [1-9][0-9]* MiB .*pipelines [0-9]+ alive \([1-9][0-9]* made" <<<"$out" \
        || { echo "FAIL the GPU VA layer reported no memory, or no pipelines made"; grep "rebox gpuva" <<<"$out" | head; exit 1; }
    echo "ok   the GPU VA layer reports the Linux side's memory and the pipelines made through it"
fi

# Negative control: same shader with the cache entry gone must fail when
# fallback shaders are off...
echo "== fake.bin, no cache entry"
rm "$work/cache/$(hash "$work/fake.bin").dxil"
REBOX_SHADER_FALLBACK=0 "$WINE" "$here/build/smoke.exe" "Z:${work//\//\\}\\fake.bin" nocache

# ...and succeed through the fallback shaders by default.
echo "== fake.bin, no cache entry, fallback shaders"
# This run also counts proxy slot calls (REBOX_SLOT_STATS).
out=$(REBOX_FRAME_TRACE=0-1000 REBOX_SLOT_STATS=1 REBOX_SKIP_ASSERTS=1 REBOX_STACK_SAMPLE=1,3 REBOX_WRITE_WATCH=7e0000008:8 "$WINE" "$here/build/smoke.exe" "Z:${work//\//\\}\\fake.bin" 2>&1) || { echo "$out"; exit 1; }
echo "$out"
grep -q "uses fallback shaders" <<<"$out" || { echo "FAIL fallback shaders were not used"; exit 1; }
grep -Eq "slot stats GraphicsCommandList:.* 12c?=" <<<"$out" || { echo "FAIL slot stats lack the draw (list slot 12)"; exit 1; }
grep -Eq "slot stats Device:.* 100T=1" <<<"$out" || { echo "FAIL slot stats lack the trapped device slot 100"; exit 1; }
echo "ok   slot stats count forwarded, custom and trapped slots"
grep -q "title: rebox smoke boot line" <<<"$out" && grep -q "title: second line" <<<"$out" \
    || { echo "FAIL the title's OutputDebugString text isn't in the log"; exit 1; }
echo "ok   the title's OutputDebugString text is in the log, a line each"
grep -q "network: looking up example.com blocked" <<<"$out" || { echo "FAIL a blocked lookup is not logged"; exit 1; }
echo "ok   a blocked network attempt is logged"
grep -q "a shader view of a depth buffer as format 56 is format 46, plane 0" <<<"$out" \
    && grep -q "a shader view of a depth buffer as format 62 is format 47, plane 1" <<<"$out" \
    && grep -q "an unordered access view of a 256x144 texture (format 45) without unordered access here: a null view" <<<"$out" \
    || { echo "FAIL views of a D16S8 depth buffer are not the desktop's plane views and a null UAV"; exit 1; }
echo "ok   views of a D16S8 depth buffer: depth and stencil plane formats, a null UAV"
grep -q "ok   an assertion (int3) in the exe was stepped over" <<<"$out" && grep -q "assertion 1 skipped at smoke.exe+" <<<"$out" \
    || { echo "FAIL REBOX_SKIP_ASSERTS did not step over the exe's int3"; exit 1; }
echo "ok   REBOX_SKIP_ASSERTS steps over the exe's assertions and logs them"
# REBOX_FRAME_TRACE: a line per console stage compile, naming the dispatch or draw that asked; and
# the census's count of what each kind of stage came to.
grep -Eq "d3d12_xs: trace [0-9]+ #[0-9]+ \[[0-9a-f]+\] compute [0-9a-f]{16} stage 0{16}: ok \| dispatch " <<<"$out" \
    && grep -Eq "d3d12_xs: trace [0-9]+ #[0-9]+ \[[0-9a-f]+\] (vertex|pixel) [0-9a-f]{16} stage [0-9a-f]{16}: ok \| (draw|indexed) [0-9]+ [0-9]+ 0 rt [0-9]+x[0-9]+ f[0-9]+" <<<"$out" \
    && grep -Eq "console compiles since the last census: compute: ok [0-9]+" <<<"$out" \
    || { echo "FAIL REBOX_FRAME_TRACE traced no dispatch and draw, or the census counted no compiles"; exit 1; }
echo "ok   REBOX_FRAME_TRACE traces each stage compile with its dispatch or draw"
# Ray tracing list methods are ours, not forwarded: a state object that isn't vkd3d-proton's must
# never reach it (SetPipelineState1/DispatchRays at 120/121, and 122/123 from SDK 0x20004), and
# acceleration structure copies (119, 121) are skipped.
grep -Eq "slot stats GraphicsCommandList:.* 119c=.* 120c=.* 121c=" <<<"$out" \
    && grep -Eq "slot stats GraphicsCommandList \(SDK 0x20004\):.* 121c=.* 122c=.* 123c=" <<<"$out" \
    || { echo "FAIL ray tracing list slots are forwarded, not handled"; exit 1; }
echo "ok   ray tracing list slots are handled by d3d12_xs on both layouts"
# xmem's fault reporter (test_crash_report): module+RVA, target, GPU window, stack.
grep -Eq "first-chance access violation \(0xc0000005\) at smoke.exe\+0x" <<<"$out" \
    && grep -Eq "  reading 0x10\s*$" <<<"$out" && grep -q "reading 0x20ff00000040 (inside the GPU window" <<<"$out" \
    && grep -Eq "  #1  smoke.exe\+0x" <<<"$out" || { echo "FAIL fault report missing or incomplete"; exit 1; }
# The sampling thread's own lines: other threads log in between.
sampler=$(grep -m1 -o "^\[rebox [0-9a-f]*\] xmem: stack sample 1 " <<<"$out" | grep -o "^\[rebox [0-9a-f]*\]")
[ -n "$sampler" ] && grep -F -A3 "$sampler xmem: stack sample 1 " <<<"$(grep -F "$sampler" <<<"$out")" | grep -q "xmem: thread " \
    || { echo "FAIL REBOX_STACK_SAMPLE logged no thread stacks"; exit 1; }
echo "ok   REBOX_STACK_SAMPLE logs every thread's stack from a copy, without faulting"
grep -q "write watch: smoke.exe+0x[0-9a-f]* writes 0x7e0000008" <<<"$out" || { echo "FAIL REBOX_WRITE_WATCH logged no writer"; exit 1; }
echo "ok   REBOX_WRITE_WATCH names the instruction that writes a watched range"
echo "ok   fault reports name module+RVA, the target, the GPU window and the stack"

# REBOX_FAST leaves programs out by their names; REBOX_DEBUG_VIEW shows a drawn target in place of
# the plane. Every test's draws count towards the first present: the largest target drawn is a tie
# of 64x48 ones, the first bound wins, the console draw's (a red triangle over all of it), so frame 1
# is red where the plane is orange; frame 2 had no draws, and shows the plane.
echo "== fake.bin, REBOX_FAST=1 and REBOX_DEBUG_VIEW=auto"
out=$(REBOX_FAST=1 REBOX_DEBUG_VIEW=auto "$WINE" "$here/build/smoke.exe" "Z:${work//\//\\}\\fake.bin" 2>&1) \
    || { echo "$out" | grep -E "FAIL|DEBUG_VIEW|REBOX_FAST" | head -40; exit 1; }
grep -q "ok   REBOX_FAST=1 leaves out a ray tracing program and keeps a G-buffer one" <<<"$out" \
    && grep -q "REBOX_FAST: leaving out programs in rays,hair,volumetrics,reflections,particles" <<<"$out" \
    || { echo "FAIL REBOX_FAST did not pick programs by name"; grep -E "REBOX_FAST|program" <<<"$out" | head; exit 1; }
echo "ok   REBOX_FAST leaves out programs by the names their containers carry"
grep -Eq "REBOX_DEBUG_VIEW=auto at present 1: showing the largest render target drawn, colour 64x48 format 28" <<<"$out" \
    && [[ $(od -An -tu1 -j54 -N3 "$work/frames/frame_1.bmp" | tr -s ' ') == " 0 0 255" ]] \
    && grep -q "ok   presented frame 2 has the cleared pixels" <<<"$out" \
    || { echo "FAIL REBOX_DEBUG_VIEW did not show the drawn target"; grep -E "DEBUG_VIEW|presented frame" <<<"$out" | head
         od -An -tu1 -j54 -N4 "$work/frames/frame_1.bmp"; exit 1; }
echo "ok   REBOX_DEBUG_VIEW picks the drawn target and shows its pixels"
# The copy made when the chosen target's pass ends (test_debug_view_snapshot): frame 5 the target as
# it is at the present, frame 6 the copy (blue), not the target as a later clear left it (red).
echo "== fake.bin, REBOX_DEBUG_VIEW=32x16"
out=$(REBOX_DEBUG_VIEW=32x16 REBOX_DUMP_FRAMES_AT=1,2,3,4,5,6 "$WINE" "$here/build/smoke.exe" "Z:${work//\//\\}\\fake.bin" 2>&1) \
    || { echo "$out" | grep -E "FAIL|DEBUG_VIEW" | head -40; exit 1; }
grep -q "ok   REBOX_DEBUG_VIEW shows its copy from when its pass ended" <<<"$out" \
    || { echo "FAIL REBOX_DEBUG_VIEW did not show the copy"; grep -E "DEBUG_VIEW" <<<"$out" | head; exit 1; }
echo "ok   REBOX_DEBUG_VIEW copies the target when its pass ends and shows the copy"

# REBOX_QUARANTINE skips the dispatches of compute programs a file names by the id their compile
# logs (the dispatch test's program here; that test's checks then fail, which is not this one's business).
echo "== fake.bin, REBOX_QUARANTINE"
printf '%s\n' '# programs earlier runs lost the device after' 'a7830803eec28b79  # the dispatch test' > "$work/quarantine.txt"
# That program is a short straight-line pass (console_init_pass), which is kept by default; with
# REBOX_KEEP_INIT_PASSES=0 it is skipped as any other.
out=$(REBOX_KEEP_INIT_PASSES=0 REBOX_QUARANTINE="$work/quarantine.txt" "$WINE" "$here/build/smoke.exe" "Z:${work//\//\\}\\fake.bin" 2>&1) || true
# Its first dispatch is skipped; once its output buffer is taken as indirect arguments, the same
# program is their producer and runs anyway.
grep -q "quarantine: 1 console program(s) quarantined by Z:.*quarantine.txt, skipped: a7830803eec28b79" <<<"$out" \
    && grep -q "quarantine: console compute shader a7830803eec28b79 is quarantined" <<<"$out" \
    && grep -q "FAIL.*the deferred console shader, compiled by AnyPS5 at dispatch, wrote 0 " <<<"$out" \
    || { echo "FAIL REBOX_QUARANTINE did not skip the program it names"; grep -E "quarantine|a7830803eec28b79" <<<"$out" | head; exit 1; }
echo "ok   REBOX_QUARANTINE skips the dispatches of the programs its file names"
grep -Eq "indirect producer [0-9a-f]{16} \(.*\) is quarantined \(REBOX_QUARANTINE\) but writes indirect arguments: it runs anyway" <<<"$out" \
    && grep -q "ok   a dispatch writing a buffer noted as indirect arguments is found as their producer and runs" <<<"$out" \
    || { echo "FAIL a quarantined indirect-argument producer did not run"; grep -E "producer|quarantine" <<<"$out" | head; exit 1; }
echo "ok   a quarantined program that writes indirect arguments runs anyway"
# By default the same file leaves the pass running: a reset can't hang the GPU, and skipping one lets
# later appends count on from the last frame (console_init_pass).
echo "== fake.bin, REBOX_QUARANTINE naming a short straight-line pass"
out=$(REBOX_QUARANTINE="$work/quarantine.txt" "$WINE" "$here/build/smoke.exe" "Z:${work//\//\\}\\fake.bin" 2>&1) || true
grep -q "quarantine: console compute shader a7830803eec28b79 is quarantined but is a short straight-line pass" <<<"$out" \
    && grep -q "ok   the deferred console shader, compiled by AnyPS5 at dispatch, wrote 0x1234abcd" <<<"$out" \
    && ! grep -q "FAIL.*the deferred console shader, compiled by AnyPS5 at dispatch, wrote 0 " <<<"$out" \
    || { echo "FAIL a quarantined short straight-line pass was skipped"; grep -E "quarantine|a7830803eec28b79|deferred console shader" <<<"$out" | head; exit 1; }
echo "ok   REBOX_QUARANTINE keeps a short straight-line pass running"
