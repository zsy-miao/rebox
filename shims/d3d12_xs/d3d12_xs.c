/* Replacement for the console's d3d12_xs.dll (D3D12.X), layered on
 * vkd3d-proton's d3d12.dll.
 *
 *   - forwards the standard D3D12 exports (see d3d12_xs.def);
 *   - D3D12XboxCreateDevice returns a proxy with the Xbox vtable layouts
 *     (proxy.c) over one shared vkd3d-proton device;
 *   - hands out GPU-visible memory for xmem.dll (gpu_memory.c);
 *   - hooks pipeline creation and swaps console shader bytecode for
 *     AnyPS5-translated DXIL from the shader cache. */

#define COBJMACROS
#include <initguid.h>
#include <windows.h>
#include <d3d12.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/rebox_log.h"
#include "console_compile.h"
#include "console_fssm.h"
#include "fast.h"
#include "gpu_memory.h"
#include "proxy.h"
#include "shader_cache.h"
#include "../../generated/fallback_shaders.h"

static INIT_ONCE load_once = INIT_ONCE_STATIC_INIT;
static PFN_D3D12_CREATE_DEVICE real_create_device;

static HRESULT (STDMETHODCALLTYPE *orig_create_graphics)(ID3D12Device2 *, const D3D12_GRAPHICS_PIPELINE_STATE_DESC *, REFIID, void **);
static HRESULT (STDMETHODCALLTYPE *orig_create_compute)(ID3D12Device2 *, const D3D12_COMPUTE_PIPELINE_STATE_DESC *, REFIID, void **);
static HRESULT (STDMETHODCALLTYPE *orig_create_pipeline)(ID3D12Device2 *, const D3D12_PIPELINE_STATE_STREAM_DESC *, REFIID, void **);

static BOOL CALLBACK load_d3d12(INIT_ONCE *once, void *param, void **ctx)
{
    HMODULE module = LoadLibraryA("d3d12.dll");

    if (!module)
    {
        rebox_log("d3d12_xs: failed to load d3d12.dll (is vkd3d-proton installed?)");
        return TRUE;
    }
    real_create_device = (PFN_D3D12_CREATE_DEVICE)(void *)GetProcAddress(module, "D3D12CreateDevice");
    return TRUE;
}

static const char *debugstr_guid(const GUID *g, char *buf)
{
    wsprintfA(buf, "{%08lx-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x}", g->Data1, g->Data2, g->Data3,
              g->Data4[0], g->Data4[1], g->Data4[2], g->Data4[3], g->Data4[4], g->Data4[5], g->Data4[6],
              g->Data4[7]);
    return buf;
}

/* Pipeline state streams: each subobject is { type; payload } aligned to
 * pointer size. Payload offset and total size per type come from these. */
#define SUBOBJ(name, T) \
    typedef struct { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type; T inner; } __attribute__((aligned(sizeof(void *)))) subobj_##name;
SUBOBJ(ptr, void *)
SUBOBJ(bytecode, D3D12_SHADER_BYTECODE)
SUBOBJ(stream_output, D3D12_STREAM_OUTPUT_DESC)
SUBOBJ(blend, D3D12_BLEND_DESC)
SUBOBJ(uint, UINT)
SUBOBJ(rasterizer, D3D12_RASTERIZER_DESC)
SUBOBJ(depth_stencil, D3D12_DEPTH_STENCIL_DESC)
SUBOBJ(input_layout, D3D12_INPUT_LAYOUT_DESC)
SUBOBJ(rt_formats, struct D3D12_RT_FORMAT_ARRAY) /* no typedef in mingw headers */
SUBOBJ(sample_desc, DXGI_SAMPLE_DESC)
SUBOBJ(cached_pso, D3D12_CACHED_PIPELINE_STATE)
SUBOBJ(depth_stencil1, D3D12_DEPTH_STENCIL_DESC1)
SUBOBJ(view_instancing, D3D12_VIEW_INSTANCING_DESC)
/* Newer public subobjects the mingw headers lack. Only their sizes matter
 * here; every field is 4 bytes wide (stencil masks are two UINT8s padded to
 * 4), as in the D3D12 headers:
 *   DEPTH_STENCIL_DESC2: 4 fields, 2 x DEPTH_STENCILOP_DESC1 (5 words), 1 field
 *   RASTERIZER_DESC1: 11 fields; RASTERIZER_DESC2: 10 fields */
typedef struct { UINT words[15]; } depth_stencil_desc2;
typedef struct { UINT words[11]; } rasterizer_desc1;
typedef struct { UINT words[10]; } rasterizer_desc2;
SUBOBJ(depth_stencil2, depth_stencil_desc2)
SUBOBJ(rasterizer1, rasterizer_desc1)
SUBOBJ(rasterizer2, rasterizer_desc2)
#undef SUBOBJ

#define SUBOBJECT_TYPE_DEPTH_STENCIL2 26
#define SUBOBJECT_TYPE_RASTERIZER1 27
#define SUBOBJECT_TYPE_RASTERIZER2 28

#define LAYOUT(name, stage) { offsetof(subobj_##name, inner), sizeof(subobj_##name), stage }
static const struct
{
    SIZE_T offset, size;
    const char *stage; /* non-NULL for shader bytecode subobjects */
} subobj_layouts[] =
{
    [D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE] = LAYOUT(ptr, NULL),
    [D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS] = LAYOUT(bytecode, "vs"),
    [D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS] = LAYOUT(bytecode, "ps"),
    [D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DS] = LAYOUT(bytecode, "ds"),
    [D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_HS] = LAYOUT(bytecode, "hs"),
    [D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS] = LAYOUT(bytecode, "gs"),
    [D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS] = LAYOUT(bytecode, "cs"),
    [D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_STREAM_OUTPUT] = LAYOUT(stream_output, NULL),
    [D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND] = LAYOUT(blend, NULL),
    [D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK] = LAYOUT(uint, NULL),
    [D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER] = LAYOUT(rasterizer, NULL),
    [D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL] = LAYOUT(depth_stencil, NULL),
    [D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT] = LAYOUT(input_layout, NULL),
    [D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_IB_STRIP_CUT_VALUE] = LAYOUT(uint, NULL),
    [D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY] = LAYOUT(uint, NULL),
    [D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS] = LAYOUT(rt_formats, NULL),
    [D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT] = LAYOUT(uint, NULL),
    [D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC] = LAYOUT(sample_desc, NULL),
    [D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_NODE_MASK] = LAYOUT(uint, NULL),
    [D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CACHED_PSO] = LAYOUT(cached_pso, NULL),
    [D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_FLAGS] = LAYOUT(uint, NULL),
    [D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1] = LAYOUT(depth_stencil1, NULL),
    [D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VIEW_INSTANCING] = LAYOUT(view_instancing, NULL),
    [D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS] = LAYOUT(bytecode, "as"),
    [D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS] = LAYOUT(bytecode, "ms"),
    [SUBOBJECT_TYPE_DEPTH_STENCIL2] = LAYOUT(depth_stencil2, NULL),
    [SUBOBJECT_TYPE_RASTERIZER1] = LAYOUT(rasterizer1, NULL),
    [SUBOBJECT_TYPE_RASTERIZER2] = LAYOUT(rasterizer2, NULL),
};
#undef LAYOUT

/* A console shader with no translation would fail PSO creation, which a
 * title rarely survives. Instead the whole pipeline gets fallback shaders
 * (generated/fallback_shaders.h): graphics draws nothing, compute does
 * nothing, and the title keeps running with that effect missing. They use no
 * resources, so they fit any root signature. REBOX_SHADER_FALLBACK=0
 * restores failing, to find missing shaders. Mesh pipelines get a mesh
 * shader that emits nothing and an amplification shader that launches
 * nothing (DXIL, SM 6.5). */
static const D3D12_SHADER_BYTECODE fallback_vs_bc = { fallback_vs, sizeof(fallback_vs) };
static const D3D12_SHADER_BYTECODE fallback_ps_bc = { fallback_ps, sizeof(fallback_ps) };
static const D3D12_SHADER_BYTECODE fallback_cs_bc = { fallback_cs, sizeof(fallback_cs) };
static const D3D12_SHADER_BYTECODE fallback_ms_bc = { fallback_ms, sizeof(fallback_ms) };
static const D3D12_SHADER_BYTECODE fallback_as_bc = { fallback_as, sizeof(fallback_as) };
static const D3D12_SHADER_BYTECODE no_shader;

static BOOL fallback_enabled(void)
{
    const char *env = getenv("REBOX_SHADER_FALLBACK");
    return !env || *env != '0';
}

static void note_fallback(const char *kind)
{
    static LONG count;
    LONG n = InterlockedIncrement(&count);

    if (n <= 16 || !(n & (n - 1)))
        rebox_log("d3d12_xs: %s PSO #%ld uses fallback shaders (missing translation); the effect is skipped",
                     kind, n);
}

/* Set while a graphics PSO with fallback shaders is created on this thread: vkd3d-proton (patch
 * 0018) asks ReboxFallbackPso and makes no pipeline for it (its console stages get theirs at draw
 * time; a draw without one is left out, as the fallback would draw nothing). A title made one
 * pipeline each for some 80,000 such PSOs at load. REBOX_EAGER_FALLBACK_PIPELINES=1 makes them
 * as before. */
static __thread BOOL creating_fallback;

BOOL WINAPI ReboxFallbackPso(void)
{
    static int eager = -1;
    if (eager < 0)
    {
        const char *env = getenv("REBOX_EAGER_FALLBACK_PIPELINES");
        eager = env && *env == '1';
    }
    return creating_fallback && !eager;
}

static void note_failed_fallback(const char *kind, HRESULT hr)
{
    static LONG count;
    LONG n = InterlockedIncrement(&count);

    if (n <= 16 || !(n & (n - 1)))
        rebox_log("d3d12_xs: %s PSO #%ld failed to compile (%#lx; Xbox DXIL intrinsics, or a bad translation); "
                     "retrying with fallback shaders", kind, n, hr);
}

/* A CachedPSO blob vkd3d-proton didn't make (a console PSO cache entry, say)
 * fails creation as a cache mismatch; retry once without it. */
static BOOL retry_without_cache(HRESULT hr, const D3D12_CACHED_PIPELINE_STATE *cached)
{
    static LONG logged;

    if (SUCCEEDED(hr) || !cached->pCachedBlob || !cached->CachedBlobSizeInBytes)
        return FALSE;
    if (!InterlockedExchange(&logged, 1))
        rebox_log("d3d12_xs: PSO creation with a %zu-byte CachedPSO blob failed (%#lx); retrying without it",
                     (size_t)cached->CachedBlobSizeInBytes, hr);
    return TRUE;
}

/* A shader's input signature (its container's ISG1 part, or ISGN) or output signature (OSG1, OSG5
 * or OSGN): the semantic name and index of each register that isn't a system value. Returns how
 * many registers it names. */
static UINT signature_semantics(const BYTE *dxbc, SIZE_T size, BOOL output, const char *names[32], UINT indices[32])
{
    UINT parts, count = 0;

    memset(names, 0, 32 * sizeof(*names));
    if (size < 32 || memcmp(dxbc, "DXBC", 4))
        return 0;
    parts = *(const UINT32 *)(dxbc + 28);
    for (UINT i = 0; i < parts && 32 + 4 * i + 4 <= size; i++)
    {
        UINT32 off = *(const UINT32 *)(dxbc + 32 + 4 * i), chunk;
        const BYTE *body;
        BOOL v1;
        UINT n, stride;

        if (off + 8 > size)
            continue;
        v1 = !memcmp(dxbc + off, output ? "OSG1" : "ISG1", 4) || (output && !memcmp(dxbc + off, "OSG5", 4));
        if (!v1 && memcmp(dxbc + off, output ? "OSGN" : "ISGN", 4))
            continue;
        chunk = *(const UINT32 *)(dxbc + off + 4);
        body = dxbc + off + 8;
        if (off + 8 + chunk > size || chunk < 8)
            return 0;
        n = *(const UINT32 *)body;
        stride = v1 ? 32 : 24;
        for (UINT k = 0; k < n && 8 + (k + 1) * stride <= chunk; k++)
        {
            const BYTE *e = body + 8 + k * stride + (v1 ? 4 : 0); /* ISG1/OSG1 start with the stream */
            UINT32 name = *(const UINT32 *)e, index = *(const UINT32 *)(e + 4), sv = *(const UINT32 *)(e + 8);
            UINT32 reg = *(const UINT32 *)(e + 16);
            if (sv || reg >= 32 || name >= chunk)
                continue;
            names[reg] = (const char *)body + name;
            indices[reg] = index;
            count = max(count, reg + 1);
        }
        break;
    }
    return count;
}

/* A PSO with fallback shaders whose real stages are a console vertex and pixel shader gets them
 * attached (REBOX_GRAPHICS_GUID): vkd3d-proton (patch 0006) then compiles both with AnyPS5 at
 * each draw, as compute shaders are at dispatch, and keeps the fallback pipeline for draws whose
 * compile fails. REBOX_CONSOLE_GRAPHICS=0 turns it off. */
/* Bytes per element of the vertex formats input layouts use. */
static UINT format_bytes(DXGI_FORMAT f)
{
    switch (f)
    {
    case DXGI_FORMAT_R32G32B32A32_FLOAT: case DXGI_FORMAT_R32G32B32A32_UINT: case DXGI_FORMAT_R32G32B32A32_SINT:
        return 16;
    case DXGI_FORMAT_R32G32B32_FLOAT: case DXGI_FORMAT_R32G32B32_UINT: case DXGI_FORMAT_R32G32B32_SINT:
        return 12;
    case DXGI_FORMAT_R32G32_FLOAT: case DXGI_FORMAT_R32G32_UINT: case DXGI_FORMAT_R32G32_SINT:
    case DXGI_FORMAT_R16G16B16A16_FLOAT: case DXGI_FORMAT_R16G16B16A16_UNORM: case DXGI_FORMAT_R16G16B16A16_SNORM:
    case DXGI_FORMAT_R16G16B16A16_UINT: case DXGI_FORMAT_R16G16B16A16_SINT:
        return 8;
    case DXGI_FORMAT_R8G8_UNORM: case DXGI_FORMAT_R8G8_SNORM: case DXGI_FORMAT_R8G8_UINT: case DXGI_FORMAT_R8G8_SINT:
    case DXGI_FORMAT_R16_FLOAT: case DXGI_FORMAT_R16_UNORM: case DXGI_FORMAT_R16_SNORM: case DXGI_FORMAT_R16_UINT:
    case DXGI_FORMAT_R16_SINT:
        return 2;
    case DXGI_FORMAT_R8_UNORM: case DXGI_FORMAT_R8_SNORM: case DXGI_FORMAT_R8_UINT: case DXGI_FORMAT_R8_SINT:
        return 1;
    default:
        return 4; /* R32, R16G16, R8G8B8A8, R10G10B10A2, R11G11B10 and their kinds */
    }
}

/* Which vertex shader param each pixel shader attribute reads (SPI_PS_INPUT_CNTL_n.OFFSET), as the
 * console driver links them: attribute n reads the PS input register the FSSM's table names, whose
 * semantic is some VS output register, which the VS's table says param N writes (xcl found
 * A UE5 title's Slate PS reads COLOR0 as attr0 and TEXCOORD1 as attr1, so taking attr n = param n
 * sampled every UI texture at the wrong coordinates). Left as attr n = param n where a table or
 * signature is missing. */
static void link_interpolants(const D3D12_GRAPHICS_PIPELINE_STATE_DESC *desc, void *vs_blob, void *ps_blob, BOOL watched)
{
    struct rebox_deferred_graphics *vs = (void *)((struct rebox_deferred_shader *)vs_blob + 1);
    struct rebox_deferred_graphics *ps = (void *)((struct rebox_deferred_shader *)ps_blob + 1);
    const char *in_names[32], *out_names[32];
    UINT in_indices[32], out_indices[32], linked = 0;
    static LONG shown;
    char line[256];
    int at = 0;

    BOOL in_ok = FALSE, out_ok = FALSE;
    if (!ps->register_count || !vs->register_count
            || !(in_ok = signature_semantics(desc->PS.pShaderBytecode, desc->PS.BytecodeLength, FALSE, in_names, in_indices))
            || !(out_ok = signature_semantics(desc->VS.pShaderBytecode, desc->VS.BytecodeLength, TRUE, out_names, out_indices)))
    {
        if (watched)
            rebox_log("d3d12_xs: graphics PSO interpolants not linked: pixel stage reads %u, vertex stage exports %u, "
                      "pixel signature %s, vertex signature %s", ps->register_count, vs->register_count,
                      in_ok ? "read" : "not read", out_ok ? "read" : "not read");
        return;
    }
    for (UINT n = 0; n < ps->register_count; n++)
    {
        UINT reg = ps->registers[n], out = 32, param = 32;
        if (reg < 32 && in_names[reg])
            for (out = 0; out < 32; out++)
                if (out_names[out] && !_stricmp(out_names[out], in_names[reg]) && out_indices[out] == in_indices[reg])
                    break;
        for (param = 0; out < 32 && param < vs->register_count && vs->registers[param] != out; param++)
            ;
        if (out < 32 && param < vs->register_count)
        {
            ps->interpolators[n] = param | (ps->flat[n] ? 1u << 10 : 0);
            linked++;
        }
        if (at < 200)
            at += snprintf(line + at, sizeof(line) - at, " attr%u=%s%u->param%d%s", n, reg < 32 && in_names[reg]
                           ? in_names[reg] : "?", reg < 32 ? in_indices[reg] : 0, out < 32 && param < vs->register_count
                           ? (int)param : -1, ps->flat[n] ? " flat" : "");
    }
    ps->interpolator_count = max(ps->interpolator_count, ps->register_count);
    line[min(at, (int)sizeof(line) - 1)] = 0;
    if (watched || InterlockedIncrement(&shown) <= 8)
    {
        rebox_log("d3d12_xs: graphics PSO interpolants: %u of %u linked by semantic:%s", linked,
                     ps->register_count, line);
        if (watched)
            for (UINT n = 0; n < ps->interpolator_count && n < 32; n++)
                rebox_log("d3d12_xs:   SPI_PS_INPUT_CNTL_%u = %#x (VS exports %u params)", n, ps->interpolators[n],
                          vs->register_count);
    }
}

/* first: the first stage, in desc->VS: 0x1 a vertex shader, 0x80 a mesh shader (a mesh pipeline:
 * no input layout, and the mesh shader's output table and signature link the interpolants). */
static void attach_console_graphics(const D3D12_GRAPHICS_PIPELINE_STATE_DESC *desc, void *pso, UINT32 first)
{
    static LONG attached, refused_vs, refused_ps, refused_other;
    struct rebox_graphics *g;
    void *vs = NULL, *ps = NULL;
    SIZE_T vs_size = 0, ps_size = 0;
    const char *env = getenv("REBOX_CONSOLE_GRAPHICS"), *first_name = first == 0x80 ? "mesh" : "vertex";

    if ((env && *env == '0') || !pso)
        return;
    /* No pixel shader: a depth-only pass (a depth prepass, shadow maps); only the first stage is
     * deferred and the pipeline keeps no fragment stage (patch 0014). */
    if (!desc->VS.BytecodeLength || desc->HS.BytecodeLength
            || desc->DS.BytecodeLength || desc->GS.BytecodeLength || desc->InputLayout.NumElements > ARRAYSIZE(g->elements))
    {
        LONG n = InterlockedIncrement(&refused_other);
        if (!(n & (n - 1)))
            rebox_log("d3d12_xs: graphics PSOs not deferred: %ld without both a %s and a pixel shader, or with "
                         "tessellation or geometry stages (this one: %s %zu, pixel %zu, hull %zu, domain %zu, geometry "
                         "%zu bytes, %u input elements)", n, first_name, first_name, (size_t)desc->VS.BytecodeLength,
                         (size_t)desc->PS.BytecodeLength, (size_t)desc->HS.BytecodeLength,
                         (size_t)desc->DS.BytecodeLength, (size_t)desc->GS.BytecodeLength,
                         desc->InputLayout.NumElements);
        return;
    }
    if (!(vs = console_fssm_graphics(desc->VS.pShaderBytecode, desc->VS.BytecodeLength, first, &vs_size))
            || (desc->PS.BytecodeLength
                && !(ps = console_fssm_graphics(desc->PS.pShaderBytecode, desc->PS.BytecodeLength, 0x10, &ps_size)))
            || !(g = calloc(1, sizeof(*g) + vs_size + ps_size)))
    {
        LONG n = InterlockedIncrement(vs ? &refused_ps : &refused_vs);
        if (!(n & (n - 1)))
            rebox_log("d3d12_xs: graphics PSOs not deferred: %ld whose %s shader isn't one we can read", n,
                         vs ? "pixel" : first_name);
        free(vs);
        free(ps);
        return;
    }
    g->magic = REBOX_GRAPHICS_MAGIC;
    g->version = 1;
    g->size = (UINT32)(sizeof(*g) + vs_size + ps_size);
    g->vs_offset = sizeof(*g);
    g->vs_size = (UINT32)vs_size;
    g->ps_offset = (UINT32)(sizeof(*g) + vs_size);
    g->ps_size = (UINT32)ps_size;
    /* Element r of the attached layout is the one the fetch puts in input r: the input layout
     * element whose semantic is the vertex shader's input register r (the console driver builds
     * the fetch by matching them; a UE5 title's inputs aren't in the layout's order, e.g. ATTRIBUTE0
     * then ATTRIBUTE13). Offsets are resolved here, D3D12_APPEND_ALIGNED_ELEMENT by the layout's own
     * order (patch 0006 resolves them only for elements in order). Without a signature, layout order. */
    if (first == 0x1)
    {
        const char *names[32];
        UINT indices[32], regs = signature_semantics(desc->VS.pShaderBytecode, desc->VS.BytecodeLength, FALSE, names, indices);
        UINT offsets[D3D12_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT] = { 0 }, resolved[ARRAYSIZE(g->elements)];
        for (UINT i = 0; i < desc->InputLayout.NumElements; i++)
        {
            const D3D12_INPUT_ELEMENT_DESC *e = &desc->InputLayout.pInputElementDescs[i];
            UINT slot = min(e->InputSlot, D3D12_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT - 1);
            resolved[i] = e->AlignedByteOffset == D3D12_APPEND_ALIGNED_ELEMENT ? offsets[slot] : e->AlignedByteOffset;
            offsets[slot] = resolved[i] + format_bytes(e->Format);
        }
        g->element_count = regs ? regs : desc->InputLayout.NumElements;
        for (UINT r = 0; r < g->element_count; r++)
        {
            UINT i = r;
            if (regs)
            {
                for (i = 0; i < desc->InputLayout.NumElements; i++)
                    if (names[r] && !_stricmp(desc->InputLayout.pInputElementDescs[i].SemanticName, names[r])
                            && desc->InputLayout.pInputElementDescs[i].SemanticIndex == indices[r])
                        break;
            }
            if (i < desc->InputLayout.NumElements)
            {
                const D3D12_INPUT_ELEMENT_DESC *e = &desc->InputLayout.pInputElementDescs[i];
                g->elements[r] = (struct rebox_graphics_element){ e->InputSlot, e->Format, resolved[i],
                        e->InputSlotClass == D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA };
            }
            else /* no element: no attribute (patch 0006 skips DXGI_FORMAT_UNKNOWN) */
            {
                static LONG logged;
                g->elements[r] = (struct rebox_graphics_element){ 0, DXGI_FORMAT_UNKNOWN, 0, 0 };
                if (InterlockedIncrement(&logged) <= 16)
                {
                    char layout[300];
                    int at = 0;
                    for (UINT k = 0; k < desc->InputLayout.NumElements && at < 260; k++)
                        at += snprintf(layout + at, sizeof(layout) - at, " %s%u",
                                       desc->InputLayout.pInputElementDescs[k].SemanticName,
                                       desc->InputLayout.pInputElementDescs[k].SemanticIndex);
                    layout[at] = 0;
                    rebox_log("d3d12_xs: graphics PSO: vertex input %u (%s%u) has no input layout element; it reads "
                              "zero (layout:%s)", r, names[r] ? names[r] : "?", indices[r], layout);
                }
            }
        }
    }
    /* The census and trace name stages by id; REBOX_PROGRAM_NAMES logs which program each is. */
    fast_name_id(console_stage_id(vs, vs_size), desc->VS.pShaderBytecode, desc->VS.BytecodeLength);
    if (ps)
        fast_name_id(console_stage_id(ps, ps_size), desc->PS.pShaderBytecode, desc->PS.BytecodeLength);
    if (console_stage_watched(console_stage_id(vs, vs_size)))
    {
        const D3D12_RENDER_TARGET_BLEND_DESC *b = &desc->BlendState.RenderTarget[0];
        rebox_log("d3d12_xs: watched stage's PSO: %u render targets, 0 is format %u; blend %u: colour %u*src %u %u*dst, "
                  "alpha %u*src %u %u*dst, write mask %#x", desc->NumRenderTargets, desc->RTVFormats[0], b->BlendEnable,
                  b->SrcBlend, b->BlendOp, b->DestBlend, b->SrcBlendAlpha, b->BlendOpAlpha, b->DestBlendAlpha,
                  b->RenderTargetWriteMask);
    }
    if (console_stage_watched(console_stage_id(vs, vs_size)))
        for (UINT r = 0; r < g->element_count; r++)
        {
            const D3D12_INPUT_ELEMENT_DESC *e = NULL;
            for (UINT k = 0; k < desc->InputLayout.NumElements; k++)
                if (desc->InputLayout.pInputElementDescs[k].InputSlot == g->elements[r].slot
                        && desc->InputLayout.pInputElementDescs[k].Format == g->elements[r].dxgi_format)
                    e = &desc->InputLayout.pInputElementDescs[k];
            rebox_log("d3d12_xs: watched stage's PSO: vertex input %u is %s%u, format %u, slot %u, offset %u%s", r,
                      e ? e->SemanticName : "?", e ? e->SemanticIndex : 0, g->elements[r].dxgi_format, g->elements[r].slot,
                      g->elements[r].offset, g->elements[r].per_instance ? ", per instance" : "");
        }
    if (ps)
        link_interpolants(desc, vs, ps, console_stage_watched(console_stage_id(vs, vs_size))
                          || console_stage_watched(console_stage_id(ps, ps_size)));
    memcpy((BYTE *)g + g->vs_offset, vs, vs_size);
    if (ps)
        memcpy((BYTE *)g + g->ps_offset, ps, ps_size);
    if (SUCCEEDED(ID3D12PipelineState_SetPrivateData((ID3D12PipelineState *)pso, &REBOX_GRAPHICS_GUID, g->size, g))
            && !(InterlockedIncrement(&attached) & (attached - 1)))
        rebox_log("d3d12_xs: graphics PSOs deferred to draw time: %ld", attached);
    if (attached <= 4)
        rebox_log("d3d12_xs: graphics PSO: console %s shader (%u inputs) and pixel shader (%u interpolants) "
                     "deferred to draw time, %u input elements", first_name,
                     ((const struct rebox_deferred_graphics *)((const struct rebox_deferred_shader *)vs + 1))->input_count,
                     ps ? ((const struct rebox_deferred_graphics *)((const struct rebox_deferred_shader *)ps + 1))->interpolator_count : 0,
                     g->element_count);
    free(g);
    free(vs);
    free(ps);
}

static HRESULT STDMETHODCALLTYPE hook_create_graphics(ID3D12Device2 *device,
        const D3D12_GRAPHICS_PIPELINE_STATE_DESC *desc, REFIID riid, void **pso)
{
    D3D12_GRAPHICS_PIPELINE_STATE_DESC copy;
    BOOL miss = FALSE, left_out;

    HRESULT hr;

    if (!desc)
        return orig_create_graphics(device, desc, riid, pso);
    {
        static const char *const names[] = { "vs", "ps", "ds", "hs", "gs" };
        const D3D12_SHADER_BYTECODE stages[] = { desc->VS, desc->PS, desc->DS, desc->HS, desc->GS };
        shader_cache_census("graphics", names, stages, 5, &desc->CachedPSO);
    }
    copy = *desc;
    miss |= shader_cache_substitute(&desc->VS, &copy.VS, "vs") == SHADER_MISS;
    miss |= shader_cache_substitute(&desc->PS, &copy.PS, "ps") == SHADER_MISS;
    miss |= shader_cache_substitute(&desc->DS, &copy.DS, "ds") == SHADER_MISS;
    miss |= shader_cache_substitute(&desc->HS, &copy.HS, "hs") == SHADER_MISS;
    miss |= shader_cache_substitute(&desc->GS, &copy.GS, "gs") == SHADER_MISS;
    if (miss && fallback_enabled())
    {
        copy.VS = fallback_vs_bc;
        copy.PS = desc->PS.BytecodeLength ? fallback_ps_bc : no_shader;
        copy.HS = copy.DS = copy.GS = no_shader;
        memset(&copy.StreamOutput, 0, sizeof(copy.StreamOutput));
        if (copy.PrimitiveTopologyType == D3D12_PRIMITIVE_TOPOLOGY_TYPE_PATCH)
            copy.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        note_fallback("graphics");
    }
    creating_fallback = miss && fallback_enabled();
    hr = orig_create_graphics(device, &copy, riid, pso);
    if (retry_without_cache(hr, &copy.CachedPSO))
    {
        memset(&copy.CachedPSO, 0, sizeof(copy.CachedPSO));
        hr = orig_create_graphics(device, &copy, riid, pso);
    }
    creating_fallback = FALSE;
    /* REBOX_FAST (fast.c): a program left out keeps its fallbacks, so its draws do nothing. */
    left_out = miss && (fast_skip_reason(desc->VS.pShaderBytecode, desc->VS.BytecodeLength, "vs")
                        || fast_skip_reason(desc->PS.pShaderBytecode, desc->PS.BytecodeLength, "ps"));
    if (SUCCEEDED(hr) && miss && fallback_enabled() && pso && !left_out)
        attach_console_graphics(desc, *pso, 0x1);
    /* Bytecode that passed as PC's or came from the cache can still fail to compile: Xbox DXIL
     * calling @xbox.* intrinsics (XboxBvhBuilder), or a bad translation. Fallbacks then, too. */
    if (FAILED(hr) && !miss && fallback_enabled() && hr != E_OUTOFMEMORY)
    {
        copy.VS = fallback_vs_bc;
        copy.PS = desc->PS.BytecodeLength ? fallback_ps_bc : no_shader;
        copy.HS = copy.DS = copy.GS = no_shader;
        memset(&copy.StreamOutput, 0, sizeof(copy.StreamOutput));
        memset(&copy.CachedPSO, 0, sizeof(copy.CachedPSO));
        if (copy.PrimitiveTopologyType == D3D12_PRIMITIVE_TOPOLOGY_TYPE_PATCH)
            copy.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        note_failed_fallback("graphics", hr);
        creating_fallback = TRUE;
        hr = orig_create_graphics(device, &copy, riid, pso);
        creating_fallback = FALSE;
    }
    return hr;
}

/* A deferred console compute program (SCRD version 1) is named in the trace by its code's hash. */
static void name_compute(const D3D12_SHADER_BYTECODE *deferred, const D3D12_SHADER_BYTECODE *container)
{
    const struct rebox_deferred_shader *h = deferred->pShaderBytecode;
    UINT64 hash = 0xcbf29ce484222325ull;

    if (!h || deferred->BytecodeLength < sizeof(*h) || h->magic != REBOX_DEFERRED_MAGIC || h->version != 1
            || h->code_offset > deferred->BytecodeLength || h->code_dwords * 4ull > deferred->BytecodeLength - h->code_offset)
        return;
    for (UINT i = 0; i < h->code_dwords * 4u; i++)
        hash = (hash ^ ((const BYTE *)h + h->code_offset)[i]) * 0x100000001b3ull;
    fast_name_id(hash, container->pShaderBytecode, container->BytecodeLength);
}

static HRESULT STDMETHODCALLTYPE hook_create_compute(ID3D12Device2 *device,
        const D3D12_COMPUTE_PIPELINE_STATE_DESC *desc, REFIID riid, void **pso)
{
    D3D12_COMPUTE_PIPELINE_STATE_DESC copy;
    static const char *const names[] = { "cs" };
    BOOL miss;
    HRESULT hr;

    if (!desc)
        return orig_create_compute(device, desc, riid, pso);
    shader_cache_census("compute", names, &desc->CS, 1, &desc->CachedPSO);
    copy = *desc;
    /* REBOX_FAST (fast.c): a program left out gets the fallback, as a missing translation does. */
    if (fast_skip_reason(desc->CS.pShaderBytecode, desc->CS.BytecodeLength, "cs"))
        miss = TRUE;
    else if (!(miss = shader_cache_substitute(&desc->CS, &copy.CS, "cs") == SHADER_MISS))
        name_compute(&copy.CS, &desc->CS);
    if (miss && fallback_enabled())
    {
        copy.CS = fallback_cs_bc;
        note_fallback("compute");
    }
    hr = orig_create_compute(device, &copy, riid, pso);
    if (retry_without_cache(hr, &copy.CachedPSO))
    {
        memset(&copy.CachedPSO, 0, sizeof(copy.CachedPSO));
        hr = orig_create_compute(device, &copy, riid, pso);
    }
    if (FAILED(hr) && !miss && desc->CS.BytecodeLength && fallback_enabled() && hr != E_OUTOFMEMORY)
    {
        copy.CS = fallback_cs_bc;
        memset(&copy.CachedPSO, 0, sizeof(copy.CachedPSO));
        note_failed_fallback("compute", hr);
        hr = orig_create_compute(device, &copy, riid, pso);
    }
    return hr;
}

/* Puts fallback shaders in every stage of a parsed pipeline stream. */
static void stream_fallbacks(BYTE *stream, const SIZE_T *stages, UINT stage_count, SIZE_T stream_output,
        SIZE_T topology)
{
    for (UINT i = 0; i < stage_count; i++)
    {
        UINT type = *(const UINT *)(stream + stages[i]);
        D3D12_SHADER_BYTECODE *bc = (D3D12_SHADER_BYTECODE *)(stream + stages[i] + subobj_layouts[type].offset);

        if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS)
            *bc = fallback_vs_bc;
        else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS)
            *bc = bc->BytecodeLength ? fallback_ps_bc : no_shader;
        else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS)
            *bc = fallback_cs_bc;
        else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS)
            *bc = fallback_ms_bc;
        else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS)
            *bc = fallback_as_bc;
        else
            *bc = no_shader; /* HS, DS, GS */
    }
    if (stream_output)
        memset(stream + stream_output, 0, sizeof(D3D12_STREAM_OUTPUT_DESC));
    if (topology && *(UINT *)(stream + topology) == D3D12_PRIMITIVE_TOPOLOGY_TYPE_PATCH)
        *(UINT *)(stream + topology) = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
}

static HRESULT STDMETHODCALLTYPE hook_create_pipeline(ID3D12Device2 *device,
        const D3D12_PIPELINE_STATE_STREAM_DESC *desc, REFIID riid, void **pso)
{
    D3D12_PIPELINE_STATE_STREAM_DESC copy;
    SIZE_T offset = 0, stages[16], topology = 0, stream_output = 0, cached_pso = 0;
    const char *census_names[16] = { 0 };
    D3D12_SHADER_BYTECODE census_bc[16] = { 0 };
    UINT stage_count = 0;
    BOOL miss = FALSE;
    /* The title's stages and input layout, for attach_console_graphics: Unreal 5 makes every
     * graphics PSO from a stream (a UE5 sample), so without this none drew. A mesh pipeline's mesh
     * shader (the desktop's type 25 by now: proxy.c renumbers the console's) goes where the vertex
     * shader would. */
    D3D12_GRAPHICS_PIPELINE_STATE_DESC console = { 0 };
    D3D12_SHADER_BYTECODE mesh = { 0 };
    BOOL other_stages = FALSE, left_out = FALSE;
    const D3D12_BLEND_DESC *blend = NULL;
    const struct D3D12_RT_FORMAT_ARRAY *rt_formats = NULL;
    BYTE *stream;
    HRESULT hr;

    if (!desc || !desc->pPipelineStateSubobjectStream || !(stream = malloc(desc->SizeInBytes)))
        return orig_create_pipeline(device, desc, riid, pso);
    memcpy(stream, desc->pPipelineStateSubobjectStream, desc->SizeInBytes);

    while (offset < desc->SizeInBytes)
    {
        UINT type = *(const UINT *)(stream + offset);

        if (type >= ARRAYSIZE(subobj_layouts) || !subobj_layouts[type].size)
        {
            static LONG once;

            if (!InterlockedExchange(&once, 1))
                rebox_log("d3d12_xs: unknown pipeline subobject type %u, passing stream through untranslated", type);
            free(stream);
            return orig_create_pipeline(device, desc, riid, pso);
        }
        if (subobj_layouts[type].stage)
        {
            D3D12_SHADER_BYTECODE *bc = (D3D12_SHADER_BYTECODE *)(stream + offset + subobj_layouts[type].offset);
            /* A stream walked out of step (a console subobject of another size) gives a bytecode
             * that isn't one: pass the stream through rather than read through it. */
            if (bc->BytecodeLength && (!bc->pShaderBytecode || IsBadReadPtr(bc->pShaderBytecode, bc->BytecodeLength)))
            {
                static LONG once;
                if (!InterlockedExchange(&once, 1))
                    rebox_log("d3d12_xs: pipeline stream walked out of step at +%#zx (%s bytecode %p, %zu bytes); "
                                 "passing it through untranslated", (size_t)offset, subobj_layouts[type].stage,
                                 bc->pShaderBytecode, (size_t)bc->BytecodeLength);
                free(stream);
                return orig_create_pipeline(device, desc, riid, pso);
            }
            if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS)
                console.VS = *bc;
            else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS)
                console.PS = *bc;
            else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS)
                mesh = *bc;
            else if (bc->BytecodeLength)
                other_stages = TRUE;
            if (stage_count < ARRAYSIZE(stages))
            {
                census_names[stage_count] = subobj_layouts[type].stage;
                census_bc[stage_count] = *bc;
            }
            /* REBOX_FAST (fast.c): a program left out keeps its fallbacks. */
            if (fast_skip_reason(bc->pShaderBytecode, bc->BytecodeLength, subobj_layouts[type].stage))
                left_out = TRUE;
            miss |= shader_cache_substitute(bc, bc, subobj_layouts[type].stage) == SHADER_MISS;
            if (stage_count < ARRAYSIZE(stages))
                stages[stage_count++] = offset;
        }
        else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY)
            topology = offset + subobj_layouts[type].offset;
        else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_STREAM_OUTPUT)
            stream_output = offset + subobj_layouts[type].offset;
        else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CACHED_PSO)
            cached_pso = offset + subobj_layouts[type].offset;
        else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND)
            blend = (const D3D12_BLEND_DESC *)(stream + offset + subobj_layouts[type].offset);
        else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS)
            rt_formats = (const struct D3D12_RT_FORMAT_ARRAY *)(stream + offset + subobj_layouts[type].offset);
        else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT)
            console.InputLayout = *(const D3D12_INPUT_LAYOUT_DESC *)(stream + offset + subobj_layouts[type].offset);
        offset += subobj_layouts[type].size;
    }
    if (blend)
        console.BlendState = *blend;
    if (rt_formats)
    {
        console.NumRenderTargets = rt_formats->NumRenderTargets;
        memcpy(console.RTVFormats, rt_formats->RTFormats, sizeof(console.RTVFormats));
    }
    shader_cache_census("stream", census_names, census_bc, stage_count,
                        cached_pso ? (const D3D12_CACHED_PIPELINE_STATE *)(stream + cached_pso) : NULL);

    if (miss && fallback_enabled())
    {
        stream_fallbacks(stream, stages, stage_count, stream_output, topology);
        note_fallback("stream");
    }

    copy.SizeInBytes = desc->SizeInBytes;
    copy.pPipelineStateSubobjectStream = stream;
    creating_fallback = miss && fallback_enabled();
    hr = orig_create_pipeline(device, &copy, riid, pso);
    if (cached_pso && retry_without_cache(hr, (D3D12_CACHED_PIPELINE_STATE *)(stream + cached_pso)))
    {
        memset(stream + cached_pso, 0, sizeof(D3D12_CACHED_PIPELINE_STATE));
        hr = orig_create_pipeline(device, &copy, riid, pso);
    }
    creating_fallback = FALSE;
    /* A mesh pipeline: its mesh and pixel shaders, compiled at each mesh dispatch (patch 0013). One
     * with an amplification shader (other_stages) keeps its fallbacks: AnyPS5 has no task stage. */
    if (left_out && miss)
        ; /* fallbacks only: its draws do nothing */
    else if (SUCCEEDED(hr) && miss && fallback_enabled() && pso && !other_stages && mesh.BytecodeLength
            && !console.VS.BytecodeLength)
    {
        console.VS = mesh;
        console.InputLayout.NumElements = 0;
        attach_console_graphics(&console, *pso, 0x80);
    }
    else if (SUCCEEDED(hr) && miss && fallback_enabled() && pso && !other_stages)
    {
        /* The first console graphics streams' blending and targets, to check that state reaches the
         * pipelines console draws get. */
        static LONG shown;
        if (InterlockedIncrement(&shown) <= 24)
            rebox_log("d3d12_xs: console graphics stream PSO %p: RT0 blend %s (src %u dst %u op %u, alpha %u/%u/%u, "
                      "mask %#x), %u targets (first format %u)", *pso, !blend ? "absent"
                      : blend->RenderTarget[0].BlendEnable ? "on" : "off",
                      blend ? blend->RenderTarget[0].SrcBlend : 0, blend ? blend->RenderTarget[0].DestBlend : 0,
                      blend ? blend->RenderTarget[0].BlendOp : 0, blend ? blend->RenderTarget[0].SrcBlendAlpha : 0,
                      blend ? blend->RenderTarget[0].DestBlendAlpha : 0, blend ? blend->RenderTarget[0].BlendOpAlpha : 0,
                      blend ? blend->RenderTarget[0].RenderTargetWriteMask : 0, rt_formats ? rt_formats->NumRenderTargets : 0,
                      rt_formats ? rt_formats->RTFormats[0] : 0);
        attach_console_graphics(&console, *pso, 0x1);
    }
    /* As for graphics and compute PSOs: bytecode that passed as PC's can still fail to compile. */
    if (FAILED(hr) && !miss && stage_count && fallback_enabled() && hr != E_OUTOFMEMORY)
    {
        stream_fallbacks(stream, stages, stage_count, stream_output, topology);
        if (cached_pso)
            memset(stream + cached_pso, 0, sizeof(D3D12_CACHED_PIPELINE_STATE));
        note_failed_fallback("stream", hr);
        creating_fallback = TRUE;
        hr = orig_create_pipeline(device, &copy, riid, pso);
        creating_fallback = FALSE;
    }
    free(stream);
    return hr;
}

static void patch_slot(void **slot, void *hook, void **orig)
{
    DWORD old;

    if (*slot == hook)
        return;
    VirtualProtect(slot, sizeof(*slot), PAGE_READWRITE, &old);
    *orig = *slot;
    *slot = hook;
    VirtualProtect(slot, sizeof(*slot), old, &old);
}

static void install_hooks(IUnknown *unk)
{
    static SRWLOCK lock = SRWLOCK_INIT;
    ID3D12Device2 *device;
    ID3D12Device2Vtbl *vtbl;

    if (FAILED(IUnknown_QueryInterface(unk, &IID_ID3D12Device2, (void **)&device)))
    {
        rebox_log("d3d12_xs: device lacks ID3D12Device2, shader substitution disabled");
        return;
    }
    /* vkd3d-proton shares one vtable across devices, so patch it once. */
    vtbl = device->lpVtbl;
    AcquireSRWLockExclusive(&lock);
    patch_slot((void **)&vtbl->CreateGraphicsPipelineState, hook_create_graphics, (void **)&orig_create_graphics);
    patch_slot((void **)&vtbl->CreateComputePipelineState, hook_create_compute, (void **)&orig_create_compute);
    patch_slot((void **)&vtbl->CreatePipelineState, hook_create_pipeline, (void **)&orig_create_pipeline);
    ReleaseSRWLockExclusive(&lock);
    ID3D12Device2_Release(device);
}

HRESULT WINAPI D3D12CreateDevice(IUnknown *adapter, D3D_FEATURE_LEVEL level, REFIID riid, void **device)
{
    HRESULT hr;

    InitOnceExecuteOnce(&load_once, load_d3d12, NULL, NULL);
    if (!real_create_device)
        return E_FAIL;
    hr = real_create_device(adapter, level, riid, device);
    if (SUCCEEDED(hr) && device && *device)
        install_hooks(*device);
    return hr;
}

static BOOL CALLBACK create_shared_device(INIT_ONCE *once, void *param, void **ctx)
{
    ID3D12Device **device = param;
    /* 11_0 is the D3D12 minimum; capabilities beyond it are queried, not
     * requested, so this doesn't limit real GPUs. */
    HRESULT hr = D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0, &IID_ID3D12Device, (void **)device);

    if (FAILED(hr))
    {
        rebox_log("d3d12_xs: creating the shared device failed: %#lx", hr);
        *device = NULL;
    }
    return TRUE;
}

ID3D12Device *rebox_shared_device(void)
{
    static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
    static ID3D12Device *device;

    InitOnceExecuteOnce(&once, create_shared_device, &device, NULL);
    return device;
}

/* Console device creation. The adapter is ignored; vkd3d-proton picks the Vulkan device. Of
 * D3D12XBOX_CREATE_DEVICE_PARAMETERS (Version, ProcessDebugFlags, GraphicsCommandQueueRingSizeBytes,
 * GraphicsScratchMemorySizeBytes, ComputeScratchMemorySizeBytes, PreferWarp: ScarlettX) only the
 * version is used: from 0x20004 the queue has a later layout (proxy.c). Every call returns the same
 * device, as on the console. */
HRESULT WINAPI D3D12XboxCreateDevice(IUnknown *adapter, const void *params, REFIID riid, void **device)
{
    ID3D12Device *shared = rebox_shared_device();
    char buf[40];

    if (params)
    {
        const UINT32 *p = params;
        rebox_log("d3d12_xs: D3D12XboxCreateDevice(riid %s): SDK version %#x, debug flags %#x, ring %u, "
                     "scratch %u/%u bytes, prefer WARP %u", debugstr_guid(riid, buf), p[0], p[1], p[2], p[3], p[4], p[5]);
        proxy_set_sdk_version(p[0]);
    }
    else
        rebox_log("d3d12_xs: D3D12XboxCreateDevice(riid %s), no creation parameters", debugstr_guid(riid, buf));
    if (!device)
        return E_POINTER;
    *device = NULL;
    if (!shared)
        return E_FAIL;
    return proxy_query_device(shared, riid, device);
}
