# Shaders: the vkd3d-proton side

A title's shaders reach `d3d12_xs` at PSO creation. PC bytecode goes to vkd3d-proton unchanged;
console shaders are compiled by AnyPS5 at dispatch and draw time ([anyps5.md](anyps5.md)). This
page covers the vkd3d-proton patches that path rests on, and the shader cache.

## Why inside vkd3d-proton

`d3d12_xs` could record console work into vkd3d-proton's command buffers from outside
(`ID3D12GraphicsCommandListExt::GetVulkanHandle`), but vkd3d-proton supports that only outside its
own render passes; graphics draws would fight its render-pass, barrier and dynamic-state tracking.
So vkd3d-proton 3.0.1 is built from source in the flake (`.#vkd3d-proton-rebox`, mingw, static
winpthreads) with `patches/vkd3d-proton/` applied in name order. Most of what console shaders need
is already there:

- root CBVs, SRVs and UAVs are buffer device addresses in push constants on RADV;
- shader-visible heaps are descriptor buffers holding raw RADV hardware descriptors with a
  32-byte CBV/SRV/UAV stride, the console's stride;
- heap memory is host-visible, so the identity layer maps it at GPU VA == CPU VA;
- wave size can be set per shader.

## The patches

| Patch | What it does |
|---|---|
| `0001-real-descriptor-heap-va` | Exports `ReboxGetDescriptorHeapVA` from `d3d12core.dll`: the real GPU address and stride of a shader-visible heap's descriptors (descriptor buffers only, as on RADV; lavapipe has none). With `REBOX_HEAP_VA=real`, heap slot 10 returns it, and the command-list slots that take a GPU handle (root descriptor tables, Xbox 35/36; UAV clears, 53/54) map it back to vkd3d-proton's handle |
| `0002-console-pipelines` | Accepts an `SCSV` blob in place of DXBC: a 44-byte header (magic `SCSV`, version 1, Vulkan stage bit, wave size 0/32/64, workgroup size, SPIR-V offset and size, DXBC offset and size), the SPIR-V (entry point `main`, used as is, with the wave size requested) and, for graphics stages, a console DXBC container whose I/O signatures give the vertex attribute locations. Every root signature's push-constant range is widened to at least 144 bytes, so console pipelines share its layout. At each draw and dispatch with such a pipeline, the user SGPRs s0-s31 are packed from the root parameters in root order: one dword per 32-bit constant, two per root descriptor or descriptor table |
| `0003-descriptor-lookup-tables` | Per-heap descriptor lookup tables, passed at push dwords 36-39 (resource table, then sampler table). `d3d12_xs` leaves them empty: AnyPS5's bindings resolve on the CPU through the shadow heaps |
| `0004-console-heap-bindings` | Remaps an `SCSV` module's set 0 binding 0 (resource heap) and binding 1 (sampler heap) to the device's bindless sets |
| `0005-deferred-console-pipelines` | Deferred console compute pipelines (`SCRD` blobs), compiled through `d3d12_xs` at each dispatch ([anyps5.md](anyps5.md)); raises the sampler-heap limit by 256 for the heaps' hidden tail |
| `0006-deferred-console-graphics` | Deferred console vertex and pixel stages, compiled at each draw ([anyps5.md](anyps5.md)) |
| `0007-console-compute-user-sgprs` | The console's compute user-SGPR layout for deferred compute shaders: root parameters from s4, one SGPR per descriptor table (a 32-bit pointer), two per root descriptor, one per constant; s0-s3 left 0. `REBOX_CONSOLE_COMPUTE_LAYOUT=pairs` packs as `0002` does, from s0 with two per table, which the smoke test's hand-written programs use |
| `0008-console-breadcrumbs` | With `VKD3D_CONFIG=breadcrumbs`, records each console variant's key before its dispatch or draw, so a lost device's breadcrumb dump names the console shader (`d3d12_xs` logs the key with each compile). The flake builds vkd3d-proton with `enable_trace` for this |
| `0009-console-heap-sizes` | Shader-visible CBV/SRV/UAV heaps of up to 2^21 descriptors where the device uses descriptor buffers and its limits allow (D3D12 guarantees 1,000,000; the target title makes one of 1,245,440). The bindless set layouts are sized from the same maximum; elsewhere it stays at 1,000,000 |
| `0010-dxil-spirv-bodiless-function` | dxil-spirv refuses an entry point or leaf function with no body (console DXIL such as `XboxBvhBuilder`'s, made of console intrinsics) instead of reading its missing entry block; the PSO then fails and gets fallback shaders |
| `0011-console-rect-lists` | The console's rect lists (topology 17): `IASetPrimitiveTopology` takes 17 and draws it as a triangle list, except that deferred console graphics get a variant with AnyPS5's rect-list tessellation stages (`ReboxConsoleRectStages`), a static 3-point patch list, so each rectangle's fourth corner is made after the vertex shader |
| `0012-console-directly-indexed-heaps` | In the graphics layout, a root signature with `CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED` or `SAMPLER_HEAP_DIRECTLY_INDEXED` gets that heap's start (its title address) in the SGPR after its root parameters, resource heap first: a text pixel shader reads its sampler from `SamplerDescriptorHeap` at s11 after a table and two constants. Without it the load faults and the stage writes nothing |
| `0013-console-mesh-pipelines` | Deferred console mesh stages ([anyps5.md](anyps5.md)): a mesh PSO's attached mesh and pixel stages compile through `ReboxConsoleCompile` wherever `0006` compiles a draw's, which covers `DispatchMesh` and `ExecuteIndirect` with a mesh-dispatch signature, the mesh stage in place of the fallback mesh shader and required at the subgroup size it was compiled for. `DispatchMesh(x, y, z)` with such a pipeline dispatches `x * y * z` groups along X (each workgroup is one console group); an indirect one can't be flattened: groups along its Y and Z would repeat X's (logged) |
| `0014-console-graphics-without-pixel-stage` | A deferred console graphics pipeline with no pixel stage (a depth prepass, shadow maps: the mesh pipelines often have none) compiles only its first stage and keeps no fragment module |
| `0015-async-console-compute-pipelines` | A deferred console compute shader's Vulkan pipelines are made by worker threads (half the CPUs, at most 8), its dispatches skipped until one is ready; the driver's compiles, up to hundreds of milliseconds each, had stalled a title's recording threads. `REBOX_CONSOLE_ASYNC=0` makes them at the dispatch |
| `0016-async-console-graphics-pipelines` | A deferred console graphics stage pair's Vulkan pipelines are made by the same worker threads, their draws using the PSO's own (fallback) pipeline until one is ready; the driver's compiles of a world's large stages took up to seconds a draw in a title's recording threads. `REBOX_CONSOLE_ASYNC=0` makes them at the draw |
| `0017-indirect-argument-clamp` | `ExecuteIndirect` copies its argument records to scratch through a meta compute pass and bounds them before the GPU uses them: a dispatch or mesh dispatch with a dimension of `REBOX_INDIRECT_CLAMP_DIM` or more (default 2^24: a console dispatches past D3D12's 65535 in X) or more than `REBOX_INDIRECT_CLAMP_GROUPS` groups (default 2^22), or a draw of more than `REBOX_INDIRECT_CLAMP_VERTICES` vertices or indices times instances (default 2^28), runs empty, and a count buffer's count is clamped to `MaxCommandCount`. A buffer whose filling dispatch was skipped (its console shader still compiling) holds anything, often all ones: billions of workgroups, and the kernel's ring timeout lost the device. Covers plain, count-buffer and state-template (device-generated commands) signatures, not rays. The copy goes where vkd3d-proton's own argument patching goes: ahead of the list until an indirect-argument barrier, in place after one (ending a render pass). The GPU counts clamped records; the log reports the count at 1, 2, 4, ... with the last record's words. `REBOX_INDIRECT_CLAMP=0` turns it off |
| `0018-lazy-fallback-pipelines-and-compile-budget` | A graphics PSO d3d12_xs made with fallback shaders (`ReboxFallbackPso` says so while it's created) gets no Vulkan pipeline of its own, and a draw of it without a console pipeline is left out (the fallback drew nothing): a title made one pipeline each for some 80,000 PSOs at load. `REBOX_EAGER_FALLBACK_PIPELINES=1` makes them as before. Console pipelines are made only while the SPIR-V words in flight fit `REBOX_PIPELINE_WORDS_IN_FLIGHT` (default 1,500,000; 0: no limit), or nothing else is being made: the driver's compiles of a world's 1-2.4M-word modules hold GBs each. `REBOX_VARIANT_LOG=1` logs what each PSO's later console pipelines differ from its first by |
| `0030-console-first-calls-build-pipelines-at-once` | When `d3d12_xs` made a call's console compiles wait (`ReboxConsoleMustDraw`: one of a program's first calls, or a full-screen pass's; README, `REBOX_FIRST_CALLS_WAIT`), its new Vulkan pipeline is made at the draw or dispatch instead of by `0015`/`0016`'s workers, so a pass the title makes only once isn't skipped while its pipeline is built |

The smoke test covers `0001`-`0004` with hand-written SPIR-V console shaders (GLSL in
`tools/gen_fallback_shaders.sh`): a compute shader writes s0-s31 to the root UAV in s0:s1, an
`SCSV` vertex shader draws a full-screen triangle whose pixel shader writes its SGPRs, and the
table addresses equal heap slot 10's start plus the table's offset on RADV (0 on lavapipe).

## The shader cache

`d3d12_xs` looks each shader that isn't PC bytecode up in `REBOX_SHADER_CACHE` (default
`$WINEPREFIX/shader-cache`) by the 64-bit FNV-1a hash of its bytes exactly as the title passes
them: `<hash>.scsv` first, then `<hash>.dxil`. PC bytecode is a container with a `SHEX`/`SHDR`
part, or a `DXIL` part holding LLVM bitcode; `SCSV` and `SCRD` blobs pass through too. Console
compute and graphics containers become deferred shaders before the cache is asked
([anyps5.md](anyps5.md)). A miss gets fallback shaders (README), and `REBOX_DUMP_SHADERS=1`
writes the missed bytes to `<cache>/dump/<hash>.<stage>.bin`.

## Open questions

| Question | Why it matters |
|---|---|
| Does a title write descriptor bytes into heap memory itself, in console format? | Those land in the shadow heap as on the console; turning them into vkd3d-proton views needs decoding |
| GFX10.3 against GFX11 descriptor fields (formats, mip fields, V# swizzle bit) | RADV writes GFX11 descriptors for the 890M; the shadow heaps hold GFX10 ones for AnyPS5 |
| BVH builder shaders (Xbox DXIL calling `@xbox.*` intrinsics) | They aren't FSSM and need a lowering of their own; their PSOs get fallbacks |
