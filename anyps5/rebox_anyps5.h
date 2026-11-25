/* rebox_anyps5.dll: AnyPS5's RDNA -> SPIR-V recompiler behind a C API for
 * d3d12_xs (docs/anyps5.md). GPL-2.0-only, as AnyPS5. */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef REBOX_ANYPS5_BUILD
#define SA5_API __declspec(dllexport)
#else
#define SA5_API __declspec(dllimport)
#endif

/* Guest memory the recompiler may read while resolving descriptors: the
 * descriptor tables and anything they point at, at their guest addresses. */
struct sa5_region
{
    uint64_t address;
    const void *data;
    size_t size;
};

struct sa5_compute_request
{
    const uint32_t *code;       /* RDNA code, starting at the first instruction */
    uint32_t code_dwords;
    uint64_t code_address;      /* guest address of code[0] */
    uint32_t wave;              /* 32 or 64 */
    const uint32_t *user_data;  /* user SGPRs s0.. */
    uint32_t user_data_count;
    uint32_t threads[3];        /* thread-group size */
    uint32_t rsrc1, rsrc2;      /* COMPUTE_PGM_RSRC1/2: float mode, user SGPRs, TGID enables, LDS */
    const struct sa5_region *regions;
    uint32_t region_count;
    uint32_t use_cache;         /* nonzero: AnyPS5's in-memory cache (parse once per code, compile per variant) */
};

/* One resource the compiled shader binds, as AnyPS5 resolved it. */
struct sa5_binding
{
    uint32_t kind;              /* ShaderRecompiler::DescriptorKind */
    uint32_t set, binding, count;
    uint32_t word_count;        /* guest descriptor words: 4 (V#, S#) or 8 (T#) per element */
    const uint32_t *words;
    int32_t image_shape;        /* ShaderRecompiler::DescriptorImageShape, or -1 */
    uint32_t role;              /* ShaderRecompiler::DescriptorRole: 0-2 guest buffers/images/samplers */
};

/* A pixel shader with the register state the console's PSO gives it (docs/anyps5.md: a title's PS09
 * register blocks). */
struct sa5_pixel_request
{
    const uint32_t *code;
    uint32_t code_dwords;
    uint64_t code_address;
    uint32_t wave;                    /* 32 or 64 */
    const uint32_t *user_data;
    uint32_t user_data_count;
    uint32_t rsrc1, rsrc2;            /* SPI_SHADER_PGM_RSRC1/2_PS */
    uint32_t input_ena;               /* SPI_PS_INPUT_ENA (the VGPR inputs, as AnyPS5's PixelInput bits) */
    uint32_t interpolator_count;      /* SPI_PS_IN_CONTROL.NUM_INTERP */
    uint32_t interpolator_settings[32]; /* SPI_PS_INPUT_CNTL_n */
    uint32_t col_format;              /* SPI_SHADER_COL_FORMAT: a nibble per render target */
    uint32_t depth_export;            /* the shader exports depth (mrtz) */
    const struct sa5_region *regions;
    uint32_t region_count;
    uint32_t use_cache;
    uint32_t push_offset, push_size;  /* this stage's push data (graphics: one block per stage); 0, 0: 0, 128 */
};

/* A vertex shader whose first instruction calls the console's fetch shader (s_swappc_b64 into a pair
 * of user SGPRs). AnyPS5 stands in for the fetch: each input names its vertex buffer V# and the
 * VGPRs the fetch fills (v[4+4r] in a UE5 title's), and becomes a Vulkan vertex attribute
 * (sa5_result_vertex_input). */
struct sa5_vertex_input
{
    uint32_t vsharp[4];               /* the buffer: address, stride, records, format */
    uint32_t register_start, register_count;
};

struct sa5_vertex_request
{
    const uint32_t *code;
    uint32_t code_dwords;
    uint64_t code_address;
    uint32_t wave;                    /* 32 or 64 */
    const uint32_t *user_data;
    uint32_t user_data_count;
    uint32_t rsrc1;                   /* SPI_SHADER_PGM_RSRC1_VS: float mode, clamp, IEEE */
    uint32_t input_count;
    struct sa5_vertex_input inputs[32];
    uint32_t push_offset;             /* where this stage's push data goes (graphics: one block per stage) */
    const struct sa5_region *regions;
    uint32_t region_count;
    uint32_t use_cache;
    uint32_t push_size;               /* 0: 128 */
};

/* A console mesh shader: an NGG program (the hardware GS stage in fast-launch mode) that AnyPS5's
 * mesh path takes as a primitive shader over a point list with no index buffer. The console
 * launches a group's lanes with v5 = group * group_stride + lane (the code reads v5 of the first
 * lane and divides), s3 bits 24-27 the wave's index in the group; user data from s8, s0-s7 being
 * the merged stage's system SGPRs (AnyPS5 fills s2-s7; its index buffer V#, which it reads from
 * s[4:7] and never uses here, is put there). Each Vulkan workgroup is one console
 * group: workgroup X is group X. The program allocates its outputs (s_sendmsg GS_ALLOC_REQ) and
 * exports vertex n and primitive n from lane n.
 *
 * AnyPS5 reads its draw parameters from push bytes 104-127 (MeshDrawPushOffsetBytes), so the
 * stage's push data, with anything sa5_finalize_draw adds, must end by byte 104; sa5_finalize and
 * sa5_finalize_draw freeze those parameters to such a draw (no index limit, no indirect
 * arguments), and the push bytes are left free. */
struct sa5_mesh_request
{
    const uint32_t *code;
    uint32_t code_dwords;
    uint64_t code_address;
    uint32_t wave;                    /* 32 or 64 */
    const uint32_t *user_data;        /* from s0; at least 8 */
    uint32_t user_data_count;
    uint32_t rsrc1;                   /* float mode, clamp, IEEE as in SPI_SHADER_PGM_RSRC1_GS */
    uint32_t threads;                 /* lanes per group */
    uint32_t max_vertices, max_primitives;
    uint32_t group_stride;            /* what v5 holds the group times; 0: max(threads, max_vertices, max_primitives) */
    uint32_t primitive_vertices;      /* output topology: 3 triangles, 2 lines, 1 points */
    uint32_t lds_bytes;
    uint32_t clip_cntl;               /* PA_CL_VS_OUT_CNTL: the clip and cull distances pos1+ carry */
    uint32_t push_offset, push_size;  /* push_size 0: 104 - push_offset */
    const struct sa5_region *regions;
    uint32_t region_count;
    uint32_t use_cache;
};

struct sa5_result;

/* Returns 0, or -1 and the recompiler's message. AnyPS5 has no amplification (task) stage. */
SA5_API int sa5_compile_mesh(const struct sa5_mesh_request *request, struct sa5_result **result, char *error,
                             size_t error_size);

SA5_API int sa5_compile_pixel(const struct sa5_pixel_request *request, struct sa5_result **result,
                              char *error, size_t error_size);

/* Returns 0 on success; on failure, the recompiler's message in error. */
SA5_API int sa5_compile_vertex(const struct sa5_vertex_request *request, struct sa5_result **result,
                               char *error, size_t error_size);
/* Vertex results: the Vulkan vertex attributes the compiled shader reads. */
SA5_API uint32_t sa5_result_vertex_input_count(const struct sa5_result *result);
SA5_API int sa5_result_vertex_input(const struct sa5_result *result, uint32_t index, uint32_t *location,
                                    uint32_t *components, uint32_t *fetch_index);
SA5_API int sa5_compile_compute(const struct sa5_compute_request *request, struct sa5_result **result,
                                char *error, size_t error_size);
SA5_API const uint32_t *sa5_result_spirv(const struct sa5_result *result, uint32_t *word_count);
SA5_API uint32_t sa5_result_binding_count(const struct sa5_result *result);
SA5_API int sa5_result_binding(const struct sa5_result *result, uint32_t index, struct sa5_binding *binding);
SA5_API const void *sa5_result_push_constants(const struct sa5_result *result, uint32_t *size);
SA5_API uint32_t sa5_result_subgroup_size(const struct sa5_result *result);
/* AnyPS5's identity for the result's pipeline: SPIR-V hash, specialization and binding layout. */
SA5_API void sa5_result_identity(const struct sa5_result *result, uint64_t identity[2]);
/* Sizes of the host buffers AnyPS5's driver fills itself (roles FaultBuffer and Gds). */
SA5_API uint32_t sa5_fault_buffer_bytes(void);
SA5_API uint32_t sa5_gds_bytes(void);
SA5_API void sa5_result_free(struct sa5_result *result);
/* Bytes AnyPS5 holds: its result memo (APS5_RESULT_MEMO_MB bounds it) and its resource plans. */
SA5_API void sa5_memory_stats(uint64_t *memo_bytes, uint64_t *plan_bytes);
/* AnyPS5's resource plans: bytes and plans held, evicted and loaded again in all, and the budget
 * Rebox set (REBOX_APS5_PLAN_MB, default 1024 MiB; 0 when none). */
SA5_API void sa5_plan_stats(uint64_t *held_bytes, uint64_t *plans, uint64_t *evictions, uint64_t *reloads,
                            uint64_t *budget);
/* The tessellation control and evaluation stages that draw a console rect list with these vertex
 * and pixel results (AnyPS5's rect-list shaders): each 3-vertex patch is a rectangle, its fourth
 * corner made after the vertex shader. The control stage binds AnyPS5's fault buffer (sa5_finalize
 * it like other stages); the evaluation stage binds nothing. Returns 0, or -1 and a message. */
SA5_API int sa5_rect_list_stages(const struct sa5_result *vertex, const struct sa5_result *pixel,
                                 struct sa5_result **control, struct sa5_result **evaluation, char *error,
                                 size_t error_size);

/* Fitting a result to vkd3d-proton's bindings (spirv_remap.cpp). Specialization constants are
 * frozen to the result's values. Each remap moves a guest descriptor array at (set, binding) to
 * (new_set, new_binding) as a runtime array, element i becoming element heap_indices[i] there; kind
 * is the binding's DescriptorKind. Each host buffer (a role AnyPS5's driver fills itself) becomes a
 * buffer-device-address pointer read from the push block: 64-bit address push_slot at byte
 * 128 + 8 * push_slot (0-7). array_length is what OpArrayLength on it returns (its u32 count). */
struct sa5_remap
{
    uint32_t set, binding, kind;
    uint32_t new_set, new_binding;
    uint32_t count;
    const uint32_t *heap_indices;
    /* sa5_finalize_draw: element i's heap index is the draw block's index word draw_index_base + i
     * (heap_indices is unused). */
    uint32_t draw_index_base;
};

struct sa5_host_buffer
{
    uint32_t set, binding;
    uint32_t push_slot;
    uint32_t array_length;
    /* Nonzero: (set, binding) is a guest buffer array, and the buffer at push_slot is a table with
     * a 16-byte entry per element, {64-bit address, u32 length in dwords, 0}. Each element becomes
     * a pointer to its address, and OpArrayLength on it reads the length. */
    uint32_t guest_array;
    /* Where the address array starts in the push block, in bytes (0: 128). Every pushed buffer of a
     * module shares one array; graphics stages put it after their push data (e.g. 112, 240). */
    uint32_t push_base;
    /* Nonzero: the buffer's address, baked into the module as a constant; push_slot is then unused. */
    uint64_t address;
};

/* Returns 0 and the validated module in *out (free with sa5_free_words), or -1 and a message.
 * Every variable at a remap's or host buffer's (set, binding) is rewritten: AnyPS5 declares several
 * at one binding (views of the guest buffers per element type, an image table per image type).
 * OpArrayLength on a view of elements other than dwords counts its own elements. A variable the
 * module uses that is left on a descriptor set fails the call, as vkd3d-proton's layout has none of
 * AnyPS5's own; SA5_FINALIZE_PARTIAL=1 allows it (offline checks that bind only some). */
SA5_API int sa5_finalize(const struct sa5_result *result, const struct sa5_remap *remaps, uint32_t remap_count,
                         const struct sa5_host_buffer *buffers, uint32_t buffer_count, uint32_t **out,
                         uint32_t *out_count, char *error, size_t error_size);
SA5_API void sa5_free_words(uint32_t *words);
/* As sa5_finalize, but nothing that changes from draw to draw is baked in, so one module serves
 * every draw of the same shader: the push block holds, at byte draw_block_base, the address of a
 * draw block {uint64 addresses[SA5_DRAW_ADDRESSES]; uint32 indices[]} the caller fills per draw.
 * Each host buffer without a baked address is addresses[push_slot]; each remap's element i is
 * heap index indices[draw_index_base + i]. */
#define SA5_DRAW_ADDRESSES 8
/* With sa5_finalize_draw, a host buffer whose array_length is SA5_LENGTH_IN_DRAW_BLOCK has its
 * length (what OpArrayLength returns) in word 2 * SA5_DRAW_ADDRESSES + push_slot of the draw block,
 * so a buffer whose size changes from draw to draw (a flattened SRT, shader data) doesn't change
 * the module. A caller using it starts its draw_index_bases past those SA5_DRAW_ADDRESSES words. */
#define SA5_LENGTH_IN_DRAW_BLOCK 0xffffffffu
SA5_API int sa5_finalize_draw(const struct sa5_result *result, const struct sa5_remap *remaps, uint32_t remap_count,
                              const struct sa5_host_buffer *buffers, uint32_t buffer_count, uint32_t draw_block_base,
                              uint32_t **out, uint32_t *out_count, char *error, size_t error_size);
/* For tests: the rewrite of sa5_finalize_draw (sa5_finalize's with draw_block_base
 * SA5_NO_DRAW_BLOCK) on a SPIR-V module with no specialization constants to freeze. */
#define SA5_NO_DRAW_BLOCK 0xffffffffu
SA5_API int sa5_finalize_words(const uint32_t *words, uint32_t word_count, const struct sa5_remap *remaps,
                               uint32_t remap_count, const struct sa5_host_buffer *buffers, uint32_t buffer_count,
                               uint32_t draw_block_base, uint32_t **out, uint32_t *out_count, char *error,
                               size_t error_size);
/* spirv-val for Vulkan 1.2 (scalar block layout): 0, or -1 and the validator's messages. */
SA5_API int sa5_validate(const uint32_t *words, uint32_t word_count, char *error, size_t error_size);

/* The descriptor tables a compute shader reads, inferred from its code (layout.cpp, moved from
 * xsx-shader-lab's rdna-run): which user SGPRs hold 32-bit table pointers, and what each table
 * holds at which dword. Shaders form {sN, 0} addresses and s_load descriptors from fixed offsets;
 * following the loaded registers to their users gives each slot's kind. */
enum sa5_slot_kind { SA5_SLOT_TEXTURE, SA5_SLOT_STORAGE, SA5_SLOT_SAMPLER, SA5_SLOT_BUFFER, SA5_SLOT_CBUFFER };
enum sa5_slot_dim { SA5_DIM_NONE, SA5_DIM_1D, SA5_DIM_2D, SA5_DIM_3D, SA5_DIM_CUBE, SA5_DIM_1D_ARRAY,
                    SA5_DIM_2D_ARRAY, SA5_DIM_2D_MSAA, SA5_DIM_2D_MSAA_ARRAY };
struct sa5_slot
{
    uint32_t table;             /* index into table_sgprs */
    uint32_t dword;             /* within the table */
    uint32_t kind;              /* enum sa5_slot_kind */
    uint32_t dim;               /* enum sa5_slot_dim (images) */
    uint32_t size;              /* dwords: 8 (T#), 4 (V#, S#) */
    uint32_t stored;            /* written (store or atomic) */
    uint32_t formatted;         /* buffers: typed access */
    uint32_t indexed;           /* buffers: idxen (structured) */
};
struct sa5_layout
{
    uint32_t wave;
    uint32_t table_count;
    uint32_t table_sgprs[16];
    uint32_t slot_count;        /* slots filled; more may exist when slot_capacity was too small */
    uint32_t notes;             /* bit 0: unresolved descriptors, 1: dynamic table offsets, 2: undecodable code */
};
/* code starts at the first instruction. Returns 0, or -1 with a message in error. */
SA5_API int sa5_analyse(const uint32_t *code, uint32_t code_dwords, uint32_t user_sgprs, struct sa5_layout *layout,
                        struct sa5_slot *slots, uint32_t slot_capacity, char *error, size_t error_size);

#ifdef __cplusplus
}
#endif
