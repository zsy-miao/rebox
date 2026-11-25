#pragma once

#include <windows.h>

/* Deferred console shaders (docs/anyps5.md): the blob d3d12_xs hands vkd3d-proton in place of DXBC
 * for a console compute shader. vkd3d-proton (patch 0005) reads only magic, version and size, keeps
 * the blob with the PSO, and at each dispatch passes it back to ReboxConsoleCompile. */
#define REBOX_DEFERRED_MAGIC 0x44524353u /* "SCRD" */
struct rebox_deferred_shader
{
    UINT32 magic;
    UINT32 version;            /* 1 */
    UINT32 size;               /* of the whole blob */
    UINT32 stage;              /* VK_SHADER_STAGE_COMPUTE_BIT */
    UINT32 wave_size;          /* 32 or 64 */
    UINT32 workgroup_size[3];
    UINT32 rsrc1, rsrc2;       /* COMPUTE_PGM_RSRC1/2 */
    UINT32 code_offset;        /* bytes from the start of the blob, 4-aligned */
    UINT32 code_dwords;
    UINT64 code_address;       /* the code's guest address */
};

/* Graphics stages (version 2): the header above with stage VK_SHADER_STAGE_VERTEX_BIT (1),
 * FRAGMENT_BIT (0x10) or MESH_BIT_EXT (0x80), then this, then the code (console_fssm.c). rsrc2's
 * user SGPR count is the stage's. TASK_BIT_EXT (0x40), an amplification stage, is never made:
 * AnyPS5 has no task stage, and a pipeline with one keeps its fallbacks. */

/* A mesh stage's limits, from its FSSM's stage data (sa5_mesh_request). */
struct rebox_deferred_mesh
{
    UINT32 threads;              /* lanes per group */
    UINT32 max_vertices, max_primitives;
    UINT32 group_stride;         /* v5 holds the group id times this */
    UINT32 primitive_vertices;   /* 3 triangles, 2 lines, 1 points */
    UINT32 lds_bytes;
    UINT32 clip_cntl;            /* PA_CL_VS_OUT_CNTL */
};

struct rebox_deferred_graphics
{
    UINT32 input_count;          /* vertex: inputs r at v[4+4r..7+4r] */
    UINT32 input_ena;            /* pixel: SPI_PS_INPUT_ENA */
    UINT32 interpolator_count;
    UINT32 interpolators[32];    /* SPI_PS_INPUT_CNTL_n */
    UINT32 col_format;           /* SPI_SHADER_COL_FORMAT */
    UINT32 depth_export;
    /* From the FSSM's interface table (0 if it has none): pixel, the signature register each
     * `v_interp attrN` reads; vertex, the one each `exp paramN` writes. */
    UINT32 register_count;
    BYTE registers[32];
    BYTE flat[32];               /* pixel: attribute N is flat (not interpolated) */
    struct rebox_deferred_mesh mesh; /* mesh */
};

/* What d3d12_xs attaches to a graphics PSO whose stages are console vertex and pixel shaders,
 * as private data under REBOX_GRAPHICS_GUID (patch 0006 reads it at the first draw): the two
 * deferred stages and the input layout, AlignedByteOffset resolved. A mesh pipeline's (patch 0013)
 * has its mesh stage where the vertex stage goes, and no input layout. */
#define REBOX_GRAPHICS_MAGIC 0x47524353u /* "SCRG" */
struct rebox_graphics_element
{
    UINT32 slot, dxgi_format, offset, per_instance;
};
struct rebox_graphics
{
    UINT32 magic;
    UINT32 version;              /* 1 */
    UINT32 size;
    UINT32 vs_offset, vs_size, ps_offset, ps_size;
    UINT32 element_count;
    struct rebox_graphics_element elements[32];
};
/* {53435247-0001-4000-8000-736361726c74} */
DEFINE_GUID(REBOX_GRAPHICS_GUID, 0x53435247, 0x0001, 0x4000, 0x80, 0x00, 0x73, 0x63, 0x61, 0x72, 0x6c, 0x74);

/* ReboxConsoleCompile's arguments; mirrored in patch 0005 (vkd3d_rebox_compile_in/out). */
struct rebox_compile_in
{
    const void *shader;
    UINT32 shader_size;
    UINT32 user_sgpr_count;
    const UINT32 *user_sgprs;
    UINT64 heap_cpu[2];        /* CPU handle of descriptor 0 of the bound CBV/SRV/UAV and sampler heaps */
    UINT32 increment[2];
    UINT32 bindless[7][2];     /* per AnyPS5 DescriptorKind: Vulkan set and binding, set ~0u if none */
    UINT32 push_offset;        /* graphics: where this stage's 128 bytes of push data go (0 or 128; a mesh stage's, 0) */
    UINT32 rect_list;          /* graphics (patch 0011): a rect-list draw: the result is kept for ReboxConsoleRectStages */
};

struct rebox_compile_out
{
    const UINT32 *spirv;       /* cached for the process's lifetime */
    UINT32 spirv_words;
    UINT32 subgroup_size;
    UINT64 key[2];
    UINT32 push[48];
    UINT32 vertex_input_count; /* vertex: the Vulkan attributes it reads */
    UINT32 vertex_inputs[32][3]; /* location, components, input r */
};

HRESULT WINAPI ReboxConsoleCompile(const struct rebox_compile_in *in, struct rebox_compile_out *out);
/* The tessellation control and evaluation stages that draw a console rect list with two stages
 * ReboxConsoleCompile gave for a rect-list draw (AnyPS5's rect-list shaders; patch 0011). */
HRESULT WINAPI ReboxConsoleRectStages(const struct rebox_compile_out *vs, const struct rebox_compile_out *ps,
                                      struct rebox_compile_out *control, struct rebox_compile_out *evaluation);

/* Before a submission: binds the textures graphics draws read whose T#s the title wrote after
 * recording them (console_compile.c). */
void console_resolve_late(void);

/* GPU memory for one draw's data (a ring reused long after the GPU read it): its address, or 0. */
UINT64 console_draw_buffer(const void *data, SIZE_T size);

/* Draws into a display-sized target (proxy.c): the console stages they compile are noted, and
 * console_display_draws_log names them (at the first presents). */
void console_display_draw_begin(int target, UINT count, BOOL rect, BOOL indexed);
void console_display_draw_end(void);
void console_display_draws_log(void);
/* A present: the next frame's compile budget starts (console_compile.c). */
void console_frame_presented(void);
/* Brackets a draw that looks like a full-screen pass (3 or 4 vertices, one instance, no index
 * buffer): its stages wait for their first compiles longer (console_compile.c). */
void console_screen_pass(BOOL on);
/* Whether a ReboxConsoleCompile call on this thread since the last query waited for its compile
 * (cleared by asking): vkd3d-proton (patch 0030) builds its pipeline at once. */
BOOL WINAPI ReboxConsoleMustDraw(void);
/* What became of a draw, mesh dispatch or ExecuteIndirect: console_draw_watch_begin before the
 * inner call, console_draw_watch_end after it (proxy.c, for REBOX_DEBUG_VIEW's per-target counts).
 * vkd3d-proton (patch 0032) tells whether a console draw found its pipeline. */
enum
{
    DRAW_FATE_RECORDED, DRAW_FATE_STANDIN, DRAW_FATE_COMPILED, DRAW_FATE_PIPELINE_BUILDING,
    DRAW_FATE_FIRST_COMPILING, DRAW_FATE_FIRST_FAILED, DRAW_FATE_PIXEL_NOT_READY, DRAW_FATE_PLAIN, DRAW_FATES
};
void WINAPI ReboxConsoleDrawPipeline(BOOL found);
void console_draw_watch_begin(void);
UINT console_draw_watch_end(void);
/* Guest buffers (V#s with a length) the draw last watched had read as empty: not GPU memory, and
 * couldn't be made so (REBOX_DEBUG_VIEW_PROBE logs them a draw). */
UINT console_draw_watch_empty_buffers(void);
const char *console_draw_fate_name(UINT fate);
/* REBOX_FRAME_TRACE: whether this present is traced, and the draw, ExecuteIndirect or dispatch about
 * to compile its stages (op NULL: none), with its first render target (console_compile.c). */
BOOL console_trace_active(void);
void console_trace_op(const char *op, UINT a, UINT b, UINT c, UINT width, UINT height, UINT format);

/* Something a guest descriptor's binding is resolved from changed (console_compile.c's binding
 * cache): a shader-visible heap's slots or the heaps there are (textures FALSE), or which texture an
 * address names (TRUE). */
void console_bindings_changed(BOOL textures);

UINT64 console_stage_id(const void *shader, SIZE_T size);
UINT64 console_program_hash(const void *shader, SIZE_T size);
BOOL console_stage_watched(UINT64 id);
UINT64 console_last_draw_vs(void);

/* The finalized modules kept (they live for the process), and their SPIR-V's bytes. */
void console_compile_module_stats(UINT *count, UINT64 *bytes);
