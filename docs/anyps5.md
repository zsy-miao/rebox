# Console shaders with AnyPS5

Console shaders are RDNA2 machine code. Rebox compiles them to SPIR-V with AnyPS5's recompiler
(`rebox_anyps5.dll`, built from AnyPS5 `fc973e0189` with `patches/anyps5/`) when they are first
dispatched or drawn, and binds the result through vkd3d-proton ([shaders.md](shaders.md) has the
vkd3d-proton side).

## What AnyPS5 needs from a host

| | AnyPS5 |
|---|---|
| Compile input | code words, stage, user SGPR values, compute stage info (from `RSRC1`/`RSRC2`: user SGPR count, TGID enables, thread-id components, LDS size, float mode), and **guest memory**: a snapshot region or a read callback (`SrtRuntime::readMemory`) |
| Why memory | the recompiler walks the descriptor tables the SGPRs point at, resolves each T#/V#/S# and specialises on them (formats, dimensions, compare/gather) |
| Result | SPIR-V plus specialisation constants; `bindings[]` with kind, set/binding (`stage*65 + local`, set 0, or 1 for fragment) and the descriptor **by value** (4 or 8 raw words); 128-byte push data; ShaderData (16 KB: user data, buffer offsets, image/sampler metadata); a BDA page table and a fault buffer; required subgroup size |
| Per draw | a fresh resource snapshot. A cached result is patched when only V# words, user data or the flattened SRT change; new T#/S# values or V# fields that feed specialisation need a new result (possibly a new module) |

So compiles happen at the first draw or dispatch that uses a pipeline, when the real descriptors
are known, and results are cached by AnyPS5's own variant key. AnyPS5's own driver (its Vulkan
device, texture cache and descriptor decoding) isn't used: vkd3d-proton owns the device and every
resource the title created, and each binding maps to the vkd3d-proton descriptor the title made.

## The shadow heap

AnyPS5 reads descriptors as the console writes them (GFX10 layout); vkd3d-proton's heaps hold
what RADV writes (GFX11 on the 890M). `d3d12_xs` keeps a **shadow heap** per CBV/SRV/UAV and
sampler heap, in console format (`shims/d3d12_xs/shadow.c`, encoders in `console_desc.c`):

- On every descriptor write through the API (device slots 17-19, 22-24), `d3d12_xs` encodes the
  view as a GFX10 T#/V#/S# into the shadow at the same index, in the field layout AnyPS5's
  decoders read. Buffers carry their GPU address; textures a synthetic address unique per
  resource. `CopyDescriptors*` copy shadow bytes, so CPU-only heaps have shadows too. The stride
  is the increment the title gets: at least 32 bytes (16 for samplers).
- The address the title sees for a heap (heap slot 10) is the shadow's title address, in a 32-bit
  range from `0x10000000`, 64 KiB aligned (`REBOX_HEAP_VA`, README). The title's shaders truncate
  descriptor-table pointers to 32 bits, and AnyPS5 resolves them on the CPU through the read
  callback.
- Descriptors a title writes into heap memory itself land in the shadow in console format, as on
  the console.

A binding maps back to a real descriptor through a CPU hash over every shadow slot, keyed on its
console words: `ReboxShadowFind(words)` gives the vkd3d-proton CPU handle written alongside them
(shader-visible heaps first). The exports for the dispatch path and the tests are
`ReboxShadowHeap(cpu_start, &title_va, &data, &stride, &count)`, `ReboxShadowRead(title_va, out,
size)` (fails unless the range lies in one heap) and `ReboxShadowFind(words, word_count, sampler,
&cpu)`. Shadows sit in GPU-visible memory (`ReboxGpuAlloc`).

## Console containers

A console shader arrives as DXBC whose `DXIL` part is an `XBCH` header and a zlib stream (a `NOBC`
stub where LLVM bitcode would be) holding **FSSM**: GFX10.3 code plus its register setup
(`console_fssm.c`; zlib from zlib's `contrib/puff`, `shims/d3d12_xs/puff.c`). The compute header
fields were measured on 2,555 compute shaders of one title against its PS09 cache, matched by the
24-byte shader ID at 0x150:

| Offset | Field | Agreement with PS09 |
|---|---|---|
| 0x104 | float mode | 2,472 |
| 0x138 | `COMPUTE_PGM_RSRC2` | user SGPR count in all; the enables and LDS describe this code, PS09 the driver's recompile |
| 0x13C, 0x140, 0x144 | thread-group size | all |
| 0xD4 | bit 5: wave32 (clear: wave64), from americas-recomp | of 2,598 matched, the 1,464 with it set are all wave32 by `sa5_analyse`, 828 of the rest wave64 |
| 0x168 | code (u32 at 0xC0 plus 0x28) | |

`RSRC1` isn't in the header; AnyPS5 needs only its float mode and clamp bits, which PS09 has as
0x402c03xx throughout, so it is built as 0x402c0000 plus the float mode. A shorter layout (Unreal
Engine 5 titles, SDK 0x20002; its code at 0x158) has the same fields 8 bytes earlier: float mode
at 0xFC, `RSRC2` at 0x130, the thread-group size at 0x134-0x13C, the shader ID at 0x140; its wave
size comes from `sa5_analyse`, since the analysis can't tell a wave64 shader with no wave-wide
instructions from a wave32 one where the container says.

Graphics FSSMs (the shorter layout) carry less register state: the code at the u32 at 0xC0 plus
0x28 (its size in bytes at 0xC8), `SPI_SHADER_PGM_RSRC1` at 0x108, the float mode at 0xFC. The
target title's (sub-version 6 at +0x10) have the float mode at 0x104 (0xC0 in all its vertex and
pixel shaders; 0xFC holds a count there) and no `RSRC1`, which is then built as for compute, and
their code size counts the 24-byte shader ID before the code. The rest comes from the code:

- **vertex**: the first instruction calls the fetch shader (`s_swappc_b64 s[4:5], s[4:5]`) the
  console driver builds from the PSO's input layout, which leaves input r in `v[4+4r..7+4r]`; the
  inputs are the consecutive slots from v4 read before being written. No call, no inputs
  (full-screen passes). 32 user SGPRs, which `RSRC2` holds with `USER_SGPR_MSB` (bit 27).
- **pixel**: 24 user SGPRs (the prologue's `s_mov_b32 m0, s24` takes the primitive mask from the
  SGPR after them); interpolants from `v_interp_*` (flat: only `v_interp_mov`); export formats
  from `exp` (packed: FP16, else the 32-bit format of the channels written); depth from an MRTZ
  export.
- **interpolants** are linked as the console driver links them: `v_interp attrN` and
  `exp paramN` name signature registers numbered by what each shader uses. The FSSM's interface
  tables give them (xcl, GPL-2.0-only: sub-version at +0x10; pixel records at +0x130 with
  `SPI_PS_INPUT_ENA`, vertex records at +0x1D4), and `SPI_PS_INPUT_CNTL_n` is the VS param whose
  OSG1 register has the semantic of the PS's ISG1 register for attribute n. A param can hold a
  packed pair of fp16 values from two registers (the target title's: 3,566 of 8,523 pixel tables
  and 189 of 1,165 vertex tables); it links by its first.

The `FLAT_SCRATCH` setup many console shaders open with (`s_setreg_imm32_b32` or `s_setreg_b32`
to `HW_REG_FLAT_SCR_LO/HI`) becomes same-size `s_nop`s before each compile. Only the scalar
prologue is scanned, up to the first vector or memory instruction.

The console driver's own container, **FSPX** in a DXBC part `ACCC`, is read too: AMD FidelityFX's
console upscaler creates its compute passes from it, with no DXIL (format from xcl). After `FSPX`,
a version, an entry count and a size come 16-byte entries (type, offset, size, 0) from +0x10. Entry
0 holds the thread-group size (three u16 at +0x0C) and `COMPUTE_PGM_RSRC1`/`RSRC2` (+0x14, +0x18);
entry 6 the code behind a 0x24-byte header, up to its last `s_endpgm`.

## Compute

1. **PSO creation** (patch 0005). A console compute container becomes a deferred console shader,
   an `SCRD` blob (`shims/d3d12_xs/console_compile.h`): header, `RSRC1`/`RSRC2`, wave size,
   thread-group size, code address, then the code. vkd3d-proton keeps the blob with the PSO and
   creates no pipeline. `REBOX_CONSOLE_COMPUTE=0` turns this off.
2. **Dispatch.** User SGPRs are packed in the console's layout (patch 0007: root parameters from
   s4, one SGPR per descriptor table, two per root descriptor, one per constant), with descriptor
   tables pointing at the shadow heaps' title addresses (recorded at `SetDescriptorHeaps` through
   `ReboxShadowHeap`). vkd3d-proton calls `d3d12_xs`'s `ReboxConsoleCompile` with the blob, the
   SGPRs, the bound heaps and, for each AnyPS5 descriptor kind, the bindless set and binding that
   holds it.
3. **Background compile.** A program's first compile (by its blob's hash) goes to a worker
   thread, and the call fails at once (`E_PENDING`): vkd3d-proton skips the dispatch, or draws
   with the PSO's fallback. Once compiled, AnyPS5's memo and the caches here make later calls
   quick, and they compile in the calling thread; a call that would wait for the worker's lock is
   skipped too. `REBOX_CONSOLE_ASYNC=0` compiles in the calling thread every time.
4. **Compile** (`console_compile.c`). AnyPS5 compiles against the shadow heaps and `d3d12_xs`'s
   GPU memory chunks (at their GPU VAs) through its in-memory cache, which parses the code once
   and makes a variant per new set of descriptors. Its disk cache (`ANYPS5_SHADER_CACHE_DIR`)
   keeps variants and resource plans across runs, and with patch 0009 the plan builds and
   variants that failed as well, with their errors: a later run fails them at once instead of
   compiling them again. Each guest binding's elements are found in the
   bound heap's shadow, which gives heap slots. `sa5_finalize_draw` (`anyps5/spirv_remap.cpp`)
   then freezes the specialisation constants, moves each guest descriptor array onto the bindless
   set as a runtime array indexed by slots read from the dispatch's **draw block**, turns host
   buffers into BDA pointers read from that block (up to 8), and validates the module. The draw
   block's address is at push byte 128. Modules are cached by AnyPS5's pipeline identity, not the
   slots: a world's compute passes take new slots every frame, and with the slots in the module
   each made a module and a pipeline, its dispatches skipped while the pipeline was built.
5. **Bind.** vkd3d-proton makes one pipeline per key with the root signature's own layout, so its
   bindless sets are already bound. After its descriptors are updated it pushes the 192 bytes from
   the compile and marks its own root parameters dirty for the next ordinary pipeline.

How each AnyPS5 binding role is served:

| AnyPS5 binding role | Here |
|---|---|
| Guest buffers (storage buffers: every V#, constant buffers included) | a table of {address, length} built from the V# words, read through BDA (buffers' GPU VAs are device addresses). vkd3d-proton keeps CBVs as uniform-buffer descriptors, which a storage-buffer binding can't read, and V#s from memory have no heap slot. The module checks only the length, so a V# with no base (or one in the first 64 KiB) gets length 0 (`ReboxGuestBufferEntry`): with its records kept, every load through it was a scalar read of address 0, a GPU page fault |
| Guest images and samplers (set 0, `stage*65 + local`) | the vkd3d-proton descriptor at a slot where the shadow holds the same words, if that slot holds the matching kind of view (the shadow records each slot's kind), preferring such a slot of several holding those words (a single-level texture's SRV and UAV have the same T#, and a storage image found the SRV's slot). AnyPS5 may rewrite a T#'s `DST_SEL` and gives samplers point-filtered twins, so matching ignores those fields |
| Samplers in no heap (built from literals in the code, or a twin) | decoded from the S# into a real sampler in the heap's hidden tail: every shader-visible heap has 256 extra slots the title doesn't see (`GetDesc` hides them) |
| A storage image also bound as a sampled image (size queries) | an SRV of the same texture and view in the hidden tail (of the whole texture for a UAV copied from a CPU-only heap, which keeps no view descriptions) |
| FlattenedSrt (binding 61): every dword the shader's scalar loads read through user SGPRs | its words, in GPU memory shared by content |
| BDA page table (59) | the memory that exists: `ReboxGpuAlloc` chunks and the buffers the title made views of, at their GPU VAs, plus each shadow heap's title address mapped to its GPU copy, so a stray address becomes an AnyPS5 fault record instead of a GPU page fault. One table for the process (16,384 ranges, the unused ones empty at the top of the address space; AnyPS5 binary-searches it and checks its length is 4 + 8 × count words), at a fixed address, rewritten in place when what it maps changes |
| Fault buffer (60), GDS (58) | one of each for the process. The fault record and the SRT guard word (set when a speculated descriptor turns out wrong) are read at each compile, logged (the first 20 of each) and cleared. Faults come from earlier dispatches, so the log doesn't name the shader |
| ShaderData (62) | its words, as FlattenedSrt |
| A graphics stage's T# that is empty when the draw is recorded, or not a T# (patch 0020 makes it empty) | the dummy texture, and its address (of the T#s the code's image instructions use, in first-use order as AnyPS5 numbers a binding's elements, the n-th whose slot holds no T# now, traced to the `s_load_dwordx8` off a table SGPR that loaded it) noted: at the next submissions the descriptor there is read again and, once the title has written it (the target title fills a per-draw table after recording the draw, as the console GPU reads it only when it runs), its view's heap slot is written into the draw block |
| A graphics stage's V# that is empty when the draw is recorded, or not a V# (patch 0020) | an empty buffer (reads zero), and its address (the n-th of the code's `s_load_dwordx4`s off the table SGPRs whose slot holds no V# now) noted: at the next submissions the V# there is read again and its address and length are written into the stage's guest buffer table (a video shader's constants come so, with its T#s; read as zero, every frame was black) |
| Image and sampler tables (261/260/63): T#s and S#s indexed at run time, found by walking tables in guest memory | bound like guest images and samplers, each element at the heap slot holding its words; the map (64) is host data. An empty table is refused with a message |
| LDS spill (set 1), scratch (set 2) | not supported: the compile fails and the dispatch is skipped, logged once per shader |

The `SpirvTarget` lists the capabilities AnyPS5 gates on (MinLod among them:
`image_sample_d_cl` needs it), limited to what RADV and lavapipe both have.

## Graphics

- **PSO creation.** A graphics PSO whose vertex and pixel shaders are console containers gets
  the fallback shaders, and `d3d12_xs` attaches both stages as `SCRD` version 2 blobs
  (`console_fssm_graphics`) plus the input layout, as private data (`REBOX_GRAPHICS_GUID`,
  `struct rebox_graphics`). PSOs made from pipeline streams (`CreatePipelineState`) get the same
  when the stream's only stages are a vertex and a pixel shader, or a mesh and a pixel shader
  (below); amplification and tessellation pipelines keep their fallbacks.
  `REBOX_CONSOLE_GRAPHICS=0` turns this off.
- **Draw** (patch 0006). At the first draw the PSO's private data is read; its vertex buffers
  join the PSO's mask. At each draw both stages compile through `ReboxConsoleCompile`, with user
  SGPRs in the console's graphics layout (root parameters from s8, one SGPR per descriptor
  table, two per root descriptor, then the start of each heap the root signature indexes directly,
  resource then sampler, one SGPR each (patch 0012); the fetch shader's address at s[4:5]). The vertex stage goes
  through `sa5_compile_vertex`, whose attributes (location r = input layout element r) make the
  vertex input. Push data is the vertex stage's 128 bytes then the pixel stage's: AnyPS5's 112
  bytes, then the address of the stage's **draw block** (`sa5_finalize_draw`): the addresses of
  the host buffers whose contents change per draw (guest buffer table, flattened SRT, shader
  data, image table maps; up to 8), then the heap slot of every element of every remapped
  descriptor array. Draw blocks come from a 128 MiB GPU ring, which wraps onto its first blocks
  only once the GPU has finished the submissions made up to 2 s after them (else a fresh ring
  takes over, and the old one goes once the GPU is past it). Nothing that changes per draw is in
  the module, so a shader keeps one module and one pipeline when the title copies its descriptors
  to new heap slots every draw. The process-wide host buffers (fault buffer, GDS, page table) are
  baked into the modules (`sa5_host_buffer.address`).
- **Memo.** Graphics stages compile through AnyPS5's resource capture and result memo
  (`CaptureResources`, then `Recompile(request, capture)`): a draw whose captured resources match
  an earlier one's reuses that result with its per-draw data patched in. Results are held by
  pointer and their SPIR-V words made only for a new module.
- **Pipeline.** vkd3d-proton's own variant with those two modules and that vertex input, cached
  per PSO by both compiles' keys and the pipeline key. A failed compile draws with the fallback
  pipeline. `REBOX_GRAPHICS_DEBUG` (README) swaps in magenta pixel shaders or full-screen vertex
  shaders to see whether draws reach the image.

### Mesh shaders

A console mesh shader is an NGG program: the hardware GS stage in fast-launch mode, one
"vertex" per lane. What it gets and does (VIctory's and xcl's notes, checked against the 6,685
mesh shaders the target title creates, all dumped by `REBOX_DUMP_SHADERS`):

- **Inputs.** User data from s8 (s0-s7 are the merged stage's system SGPRs), laid out as a vertex
  shader's user data, so its root parameters are from s16, not s8: user-data dword i is s(8 + i)
  (the FSSM's root layout names the same dwords as a vertex or pixel stage's, VIctory's
  `HwStage::user_sgpr`; the title's mesh code reads its tables through s16 and s17).
  `d3d12_xs` shifts the call's user SGPRs by 8 for a mesh stage (`mesh_user_sgprs`);
  `REBOX_MESH_USER_DATA_S8=1` doesn't. s3 bits 24-27 hold the wave's index in
  the group; v5 holds `group * stride + lane`, the code reading the first lane's and dividing by
  the stride, `max(lanes, maximum vertices, maximum primitives)` (128 in every one of the title's:
  128 lanes, two wave64 waves, up to 128 vertices and 128 triangles).
- **Outputs.** `s_sendmsg GS_ALLOC_REQ` with m0 = vertices | primitives << 12, then lane n
  exports vertex n (`exp pos`, `exp param`) and primitive n (`exp prim`: three 10-bit vertex
  indices, bit 31 culls). No per-primitive outputs in the title.
- **Container.** The FSSM's mesh record (`console_fssm.c` has the offsets): wave size, float mode,
  LDS size, the output table (param n's OSG1 register, as a vertex shader's), `PA_CL_VS_OUT_CNTL`,
  and the group's lanes, maximum vertices and primitives and output topology. The code follows a
  24-byte shader ID.

AnyPS5's mesh path translates the PS5's primitive shaders, which read an index buffer: each
workgroup takes `primitivesPerGroup` input primitives, and lane n gets the n-th vertex index in
v5. `sa5_compile_mesh` uses it on a non-indexed point list with `stride` points per group: lane
n of workgroup g then gets v5 = g * stride + n, as from the console's fast launch, and each
Vulkan workgroup is one console group. AnyPS5 reads its draw parameters (index count, base
vertex, base instance, index format, indirect argument address) from push bytes 104-127;
`sa5_finalize_draw` freezes them to no index limit and zeros, so the mesh stage's push data is
AnyPS5's 96 bytes, then its draw block's address, and nothing per dispatch goes there.

- **PSO creation.** A pipeline stream whose stages are a console mesh shader (the desktop's 25,
  once `proxy.c` has renumbered the console's 24) and a pixel shader gets the mesh fallbacks and
  both stages attached as for vertex and pixel, the mesh stage (`SCRD` version 2, stage 0x80,
  `VK_SHADER_STAGE_MESH_BIT_EXT`) where the vertex stage goes and no input layout. The pixel
  shader's interpolants are linked to the mesh shader's params through its output table and
  OSG1, as for a vertex shader. Stage 0x40 (`TASK_BIT_EXT`) is kept for amplification shaders,
  which aren't attached: AnyPS5 has no task stage, and the title has none.
  `REBOX_MESH_PIPELINES=1` is still needed for `proxy.c` to make the title's mesh pipelines.
- **Dispatch** (patch 0013). Where patch 0006 compiles a draw's stages, a mesh pipeline's compile
  at each `DispatchMesh` and each `ExecuteIndirect` with a mesh-dispatch signature (the console's
  argument type 107, 16-byte records, made desktop `DISPATCH_MESH` by `proxy.c`). `DispatchMesh`
  counts `x * y * z` groups along X; an indirect dispatch's counts are on the GPU, so groups along
  its Y and Z would repeat X's (logged; the title's records are taken as X, Y, Z). The mesh stage is required at the subgroup size AnyPS5 compiled it for (64).

## Tests

The smoke test runs the whole path on lavapipe:

- the encoders field by field, then a hand-assembled RDNA2 shader (`s_load` of a T#, V# and S#
  from two descriptor tables, `image_sample_lz`, `buffer_store`) compiled through
  `rebox_anyps5.dll` against the shadow heaps: each binding AnyPS5 makes maps back to the
  vkd3d-proton descriptor;
- `test_console_dispatch`: a deferred PSO from RDNA2 code that stores `0x1234abcd` through the V#
  at the start of its descriptor table, dispatched and read back; again with the V# behind a
  pointer in root constants into `ReboxGpuAlloc` memory; and again from a console container
  created as an ordinary compute PSO; and a scalar loop that would run four billion times, which
  the loop guard (patch 0012) stops after 2^20 iterations with a loop-limit fault;
- `test_console_draw`: a console vertex shader that calls the fetch and exports input 0 as the
  position, scaled by a constant buffer read through descriptor table 0 (s8), and a pixel shader
  exporting red, in real containers, draw a full-screen triangle;
- `test_console_mesh_compile`: a console mesh shader (three vertices, one triangle) in a container
  of the target title's FSSM layout becomes a deferred mesh stage and compiles through
  `ReboxConsoleCompile` and `sa5_finalize_draw`. lavapipe here has no mesh shaders, so no mesh
  pipeline is made or drawn with.

The dispatch tests also pass on RADV (Radeon 890M), whose bindless sets differ from lavapipe's
(samplers 0/0, typed views and images 1/1, raw buffers and CBVs 2/0; lavapipe 1/2 for typed views
and images). `REBOX_DUMP_SPIRV=DIR` writes each finalized module.

`anyps5/layout.cpp` (behind `sa5_analyse`) infers a shader's descriptor tables and wave size from
its code; `tools/ps09.py` reads the PS09 PSO cache. `sa5_check.exe` compiles shaders through the
DLL outside a title: a built-in one, a dumped one (`CODE.bin SKIP RSRC1 RSRC2 X,Y,Z`), or the
vertex, pixel or mesh shaders listed in a plan file (`vs`/`ps`/`ms PLAN`), or a built-in mesh
program (`mesh`). `tools/ms-plan.py DUMP_DIR OUT_DIR` writes a mesh plan from the `*.ms.bin` containers
`REBOX_DUMP_SHADERS` dumps (each FSSM to `OUT_DIR`, its `FLAT_SCRATCH` setup made `s_nop`s as
`console_compile.c` does), and `SA5_DUMP_SPIRV=DIR` writes each mesh module `ms` finalizes, as it does
for `cs`. `ms` and `vs` take a draw's user SGPRs from `SA5_USER=v0,v1,...` (hex, from s0) and guest
memory from `SA5_MEMORY=ADDRESS:w0,w1,...`, `SA5_MEMORY=ADDRESS+BYTES` (that many zero bytes) or
`SA5_MEMORY=ADDRESS@FILE` (a file's bytes). `tools/ps-plan.py DUMP_DIR OUT_DIR [ID...]` writes a
pixel plan from the `*.ps.bin` containers, each shader named by the id `d3d12_xs` logs it under,
with the register state `console_fssm.c` gives it; with `--draw`, each line also gets `u=` (user
SGPRs) and `m=ADDRESS@FILE` (guest memory): a table per user SGPR the FSSM's resource records name
(VIctory's reading: kind, root type, table offset, register, dwords, user-data dword, root
parameter), each slot holding a T#, V# or S# of its kind, or the words `MEM_T`/`MEM_V` give, as a
refused draw's log shows them. `vs` with `SA5_PUSH_SIZE` finalizes each module as a graphics stage
is finalized; `SA5_REMAP=1` moves its guest descriptor arrays onto bindless arrays read through a
draw block too, as `d3d12_xs` does (`sa5_finalize_draw`). `bench THREADS PLAN [FIRST [COUNT]]`
measures compile throughput: a pixel plan's
shaders compiled and validated on that many threads at once. Compute shaders, `ps` and `bench`
compile with no cache; `SA5_USE_CACHE=1` compiles through AnyPS5's caches, the disk cache
included, as `d3d12_xs` does. Of the target title's 6,685 dumped mesh shaders, 5,538 compile, pass spirv-val
and finalize with no guest memory (`ms PLAN`, as d3d12_xs compiles them); 619 more need the
descriptors a draw's memory gives (the SRT walk), and 526 use `v_mad_legacy_f32` with clamp,
which AnyPS5's decoder accepts from patch 0010. Of 497 of those (the other 29 run past 30
minutes each; one with its clamp bits cleared runs as long without the patch), 177 now compile,
validate and finalize, 303 need the draw's descriptors, and 17 fail later (8 bindings at finalize, 8 Phi
dominance, 1 sampler pairs).

`cs PLAN [FIRST [COUNT]]` compiles the compute shaders of a compute plan (`NAME CODE SKIP RSRC1
RSRC2 X,Y,Z [WAVE]`) with no guest memory and validates them; each line also gives a hash of the
SPIR-V and the entry point's loops: how many, and how many have no conditional exit (which patch
0012's guard can't stop). `SA5_DUMP_SPIRV=DIR` writes each module there. With the guard at 2^20,
329 of 353 of the target title's compute shaders (every seventh of its PS09 cache) compile this way
and all validate; 212 have loops, 8,751 in all, and every one has a conditional exit.

`memo PLAN NAME [ITERATIONS]` times the path every draw and dispatch takes once a shader is
compiled: compute shader `NAME` of a compute plan (`NAME CODE SKIP RSRC1 RSRC2 X,Y,Z [WAVE]`),
compiled again and again through AnyPS5's caches with its descriptor tables (laid out by
`sa5_analyse`) at a new address and its buffers' V#s naming new memory each time, then its identity,
its bindings and, once per module key, `sa5_finalize`, phase by phase as `d3d12_xs` times them.
`SA5_BENCH_FIXED_TABLES=1` keeps the tables in place, `SA5_BENCH_COPIES=N` moves them through N
places, `SA5_BENCH_IMAGES=N` cycles the textures through N sets and `SA5_BENCH_SLOTS=1` makes
every call a new module. It prints a digest of each call's SPIR-V, bindings and push data, to
compare builds. Under Wine, the target title's compute shaders take 10-200 us a call there, except
its bindless ones, whose 65,536-key image tables are captured at every call (1-2.3 ms with patch
0011, 2-7 ms before).

**Per call in `d3d12_xs`.** Its compile timings (logged at powers of two of each kind's compiles)
are kept apart for title threads, whose draw or dispatch waits, and background workers, whose first
compiles of programs take seconds and once dominated the averages; waiting for the compiler lock
is a phase of its own. The guest memory regions and the BDA page table are made again only when
the shadow heaps or GPU memory chunks change (generation counters in `shadow.c` and
`gpu_memory.c`), not at every call. A shader whose bindings can't be made for a reason its next
call meets again (more descriptor elements or host buffers than `d3d12_xs` binds, a binding kind it
doesn't handle, a module it can't fit) is skipped without asking AnyPS5, except at every 64th call;
the target title's bindless compute shaders reach 24,000 table elements, and each of their
dispatches captured them all before being refused. A stage binds up to 32,768 guest descriptors
in all (the target title's image tables reach 23,725 T#s). A graphics stage AnyPS5 refuses for its code (an
instruction it doesn't handle) is remembered for the run; one refused for what its tables held at
that call (an SRT walk through memory outside the regions it was given, a descriptor that isn't
one) is retried at every 64th call instead, as a binding failure is. The regions are the shadow
heaps and up to 1,024 GPU memory chunks in all: with 64, the target title's world passed the
limit, and its mesh stages' user-data tables in the chunks left out were unreadable.

## AnyPS5 patches

Patch 0004's speculation of descriptors and table pointers it can't evaluate is on by default:
`d3d12_xs` sets `APS5_SPECULATE_DESCRIPTORS` when it loads, before anything loads AnyPS5, and
`REBOX_ANYPS5_SPECULATE=0` turns it off. A wrong guess sets the fault buffer, which `d3d12_xs`
reads back and logs. Patch 0012's loop guard is on by default too: `d3d12_xs` sets
`APS5_LOOP_GUARD=0x100000`, so every translated loop leaves after 2^20 iterations per entry
instead of running as long as data we got wrong says (a GPU hang, then a lost device). A loop
stopped this way is logged as `console shader <hash> left a loop at its limit`; `APS5_LOOP_GUARD=0`
turns the guard off and another value sets the limit. With patch 0014, `APS5_LOOP_GUARD=stages`
gives each stage its own limit instead (128 iterations in pixel programs, 256 in vertex and mesh
programs, 4096 in compute programs), and `APS5_LOOP_GUARD_PIXEL`, `_VERTEX`, `_MESH` and `_COMPUTE`
set one stage's. Patch 0020's null descriptors are on for graphics stages: `d3d12_xs` sets
`APS5_NULL_INVALID_DESCRIPTORS=graphics`, so a T# or V# that isn't one when a draw is recorded (the
target title fills a draw's tables after recording it, and until then they hold an earlier draw's
words: V#s where T#s go, constants) is bound as a null descriptor instead of refusing the stage, and
its slot is filled at submission as an empty one's is (the late T#s and V#s above).
`REBOX_ANYPS5_NULL_INVALID=0` turns it off; compute stages are refused as before. The other patches
are listed in [patches/anyps5/README.md](../patches/anyps5/README.md).

## Where AnyPS5 runs

The recompiler (`core/shader/recompiler`, without AnyPS5's driver) is built as a Windows DLL with
mingw (`.#rebox-anyps5`, from a sparse checkout of AnyPS5 and the three submodules it pins:
SPIRV-Headers, SPIRV-Tools, glslang; only the patches' `core/shader/` hunks are applied). The
guest libc is replaced by an include path and two stream functions, and winpthreads is linked
statically. `d3d12_xs` loads it in-process; its C API is `anyps5/rebox_anyps5.h`.

**Allocation.** The DLL's `operator new` and `delete` (AnyPS5, SPIRV-Tools and glslang allocate
through them) are mimalloc's (`anyps5/allocator.cpp`; mimalloc 3, MIT, linked statically from
nixpkgs' source), with per-thread heaps. With the C runtime's `malloc`, every allocation took the
process heap's single lock in Wine, and `d3d12_xs`'s compile workers spent their time queued on
it. `sa5_check bench` on 400 of the target title's pixel shaders, under Wine on 24 cores: 8 threads
take 5.0 s instead of 11.8 s, 16 threads 3.1 s instead of 11.3 s; one thread is no faster.
`malloc` itself isn't overridden (the C runtime's own blocks stay on its heap). mimalloc is built
with `MI_WIN_INIT=FLS`: its default start-up is a TLS callback that the loader calls before
mcfgthread's (the toolchain's thread runtime), and its first thread-local access there breaks
mcfgthread for the rest of the process.

**Licence.** AnyPS5 is GPL-2.0-only, and a `d3d12_xs` that loads it in-process, with the
vkd3d-proton patch that calls it, is a GPL-2 combination: Rebox is GPL-2.0-only for this reason
([LICENSE](../LICENSE)). mimalloc, linked into the DLL, is MIT.

## The PS09 cache

Titles that ship a PS09 PSO cache carry the console driver's register state for each pipeline:

- **Binding records** per shader: kind (SRV, UAV, CBV, sampler, root table), register
  (`slot_id`), a dword offset (`byte_offset`) and a packed word: descriptor dword count in bits
  0-7, user SGPR in bits 8-15, register space from bit 16 (TomBox). Against the tables
  `sa5_analyse` infers from the code, 68,991 of 73,512 inferred slots (94%) have a record at
  exactly that user SGPR and dword offset, with matching kinds; the misses are mostly samplers.
  So the console maps registers to SGPRs per shader, not in root-parameter order.
- **Graphics PSO records** (0x70 bytes): head word 2 is a register block, word 3 the input
  signature; seven stage slots (0 pixel, 1 vertex, 2 NGG) each pair a 20-byte blob with that
  stage's binding records. In the 388-byte register blocks, checked against the dumped code:

  | Word | Register (GFX10) | Agrees with the code |
  |---|---|---|
  | 14, 15 | `SPI_SHADER_PGM_RSRC1/2_VS` | all 1,328 dumped VS |
  | 19, 20 | `SPI_SHADER_PGM_RSRC1/2_PS` | all 9,519 dumped PS |
  | 24, 25 | `SPI_SHADER_PGM_RSRC1/2_GS` of NGG pipelines | all 7,498 dumped NGG shaders |
  | 22 | `SPI_PS_INPUT_ENA` | not checked |
  | 23 | `SPI_PS_IN_CONTROL.NUM_INTERP` | 1,149 |
  | 28 | `SPI_SHADER_COL_FORMAT` (a nibble per target; 4 = FP16) | 1,155 |
  | 30 | `CB_SHADER_MASK` | 1,141 |
  | 52-59 | `CB_BLENDn_CONTROL` | by value |
  | last 3 + n, padded to even | `SPI_PS_INPUT_CNTL_0..n-1` | block size = base + these words in every PSO |
  | 64-71 | `SX_MRTn_BLEND_OPT` | by value |

  Blocks of 464-520 bytes (NGG + PS) keep the `RSRC` words in place, but the pixel-output group
  sits 21 words later (`NUM_INTERP` word 44, `COL_FORMAT` word 49). `sa5_compile_pixel` builds
  AnyPS5's `ShaderPixelStageInfo` from these words.
- PS09's own code is the driver's recompile, so a deferred shader built from a container uses
  the container's code.

## Open questions

- **What a title with a PS09 cache passes at PSO creation**: the FSSM container or a PS09 blob.
  `d3d12_xs` logs each PSO's stage bytecode kinds and `CachedPSO` blob (the first PSOs, then at
  powers of two; `REBOX_DUMP_SHADERS=1` dumps them). A cache blob vkd3d-proton didn't make fails
  creation as a mismatch, so `d3d12_xs` retries without it.
- **SGPRs from the PS09 binding records.** Where a title has them, packing user SGPRs from the
  records (for each, the root parameter covering its register and space, plus the offset) would
  replace the root-order layout.
- **Speculated descriptors** are checked through the fault buffer, which is logged; recompiling
  on a mismatch isn't done.
- **Mesh shaders** compile but haven't drawn on a GPU yet. Still open: per-primitive outputs (none in the
  title; such a stage is refused); amplification shaders (no task stage in AnyPS5); the Y and Z
  group counts of indirect mesh dispatches, and what the console's 16-byte records hold beyond
  three group counts; root arguments an `ExecuteIndirect` changes per record, which never reach
  the console stages' user SGPRs (true of draws too). Geometry-shader NGG pipelines keep their
  fallbacks.
- **`SpirvTarget`** has fixed values; it should come from vkd3d-proton's physical device.
