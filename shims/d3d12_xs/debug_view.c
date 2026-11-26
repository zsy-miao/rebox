/* REBOX_DEBUG_VIEW: shows one of the title's render targets or depth buffers in place of the
 * presented plane, scaled to it, so that meshes drawn into the G-buffer or the depth buffer can be
 * seen even while lighting, exposure, tone mapping or the composite are broken.
 *
 * Every render target and depth buffer the title binds (OMSetRenderTargets, proxy.c) is kept in a
 * table, with the draws, ExecuteIndirects and mesh dispatches made while it was bound in the
 * current frame (mesh ones counted apart). At each PresentX the target to show is picked from that
 * frame's counts:
 *   REBOX_DEBUG_VIEW=auto (or 1)  the G-buffer (below) if a pass drew meshes, else the depth buffer
 *   REBOX_DEBUG_VIEW=depth        the depth buffer with the most draws (square ones, shadow maps,
 *                                 and those bound with a display-sized target, the UI's, aside)
 *   REBOX_DEBUG_VIEW=gbuffer[N]   render target N (default 0) of the pass with the most mesh draws
 *                                 (most draws when none drew meshes), display-sized ones aside
 *   REBOX_DEBUG_VIEW=largest      the largest render target drawn this frame, display-sized aside
 *   REBOX_DEBUG_VIEW=<W>x<H>[f<format>]  the most-drawn target of that size (and DXGI format)
 *   REBOX_DEBUG_VIEW=0x<address>  the texture placed at that title address
 * The pick is logged when it changes, and the frame's candidates every 60 presents. When the target
 * shown had 64 draws or more, what became of them is logged too (at most every 10 s): how many were
 * recorded with a pipeline, and why the rest were left out (a stage still compiling or failed, or
 * the pipeline still being built: vkd3d-proton patch 0032), so a target still at its clear value
 * tells draws that never reached the GPU from draws that drew nothing.
 *
 * The target is copied when the pass that drew it ends (the next OMSetRenderTargets of that list
 * that doesn't bind it again, or its Close), so a later pass reusing its memory doesn't hide it, and
 * the copy is shown. A pass whose draws into it were all left out (stages still compiling) isn't
 * copied, nor is one that drew nothing into it (a clear, a rebind) once a copy holds draws: the copy
 * shown is the last one with draws in it, from however many presents ago (the log line of each
 * dumped frame says which). REBOX_DEBUG_VIEW_SNAPSHOT=0 reads the target itself at the present.
 *   REBOX_DEBUG_VIEW_AFTER=<n>      copy it after the n-th draw into it recorded with a pipeline in
 *                                   each frame instead (pass ends aren't copied): partial results
 *   REBOX_DEBUG_VIEW_TEST=1|copy    a self-test of the readout: the first time a copy is made, a
 *                                   pattern (below) is cleared into the title's target just before
 *                                   it (1; with SNAPSHOT=0 in place of it, read at the present), or
 *                                   into the copy just after it (copy), and the frame that shows it
 *                                   is dumped (REBOX_DUMP_FRAMES). An 8x8 grid: colour targets
 *                                   (0.5, 0, 0.5) with every other cell (x+1)/8, (y+1)/8, 1 (cell x,
 *                                   y); depth 0 (the far plane, dark blue) with every other cell
 *                                   10^-((x+y)/4) (grey, lighter at the top left)
 *   REBOX_DEBUG_VIEW_SCALE=<f>      colour multiplier (default 1); float formats also roll off as
 *                                   x / (1 + x)
 *   REBOX_DEBUG_VIEW_CHANNEL=r|g|b|a  one channel as grey (default the colour)
 *   REBOX_DEBUG_VIEW_DEPTH=forward  depth with the far plane at 1 (default reversed: far at 0)
 *   REBOX_DEBUG_VIEW_DECADES=<n>    depth shown on a log scale over n decades (default 4)
 *   REBOX_DEBUG_VIEW_PROBE=<n>      queries around the draws into the target shown in n frames
 *                                   (default 2, 0 off): see the probe below
 * Depth is white near the camera and darker with distance; texels still at the clear value are
 * dark blue. Frame dumps (REBOX_DUMP_FRAMES) show the view too. */

#define COBJMACROS
#define WIDL_C_INLINE_WRAPPERS
#include <windows.h>
#include <d3d12.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/rebox_log.h"
#include "../../generated/blit_shaders.h"
#include "console_compile.h"
#include "debug_view.h"
#include "gpu_memory.h"
#include "present.h"
#include "title_textures.h"

enum select { SEL_AUTO, SEL_DEPTH, SEL_GBUFFER, SEL_LARGEST, SEL_SIZE, SEL_ADDRESS };

static int enabled = -1;
static enum select select_kind;
static UINT select_slot, select_w, select_h, select_format;
static UINT64 select_address;
static float colour_scale = 1.0f, decades = 4.0f;
static UINT channel;
static BOOL depth_forward, snapshots = TRUE;
static char select_text[64];
static LONG after_draws;  /* REBOX_DEBUG_VIEW_AFTER, 0: at pass ends */
enum test_mode { TEST_OFF, TEST_TARGET, TEST_COPY };
static struct
{
    enum test_mode mode;
    volatile LONG state;  /* 0 pending, 1 written, 2 shown */
    LONG written;         /* the frame it was written in */
} test;

#define TARGETS 512
#define EVICT_PRESENTS 8

struct target
{
    ID3D12Resource *res;  /* referenced while in the table */
    D3D12_RESOURCE_DESC desc;
    UINT64 address;
    BOOL depth, display;
    volatile LONG draws, mesh, binds;  /* this frame */
    volatile LONG fates[DRAW_FATES];   /* this frame's draws by what became of them (console_compile.h) */
    LONG seen;            /* the last present it was bound in */
    int partners[8];      /* as render target 0: 1 + the other targets of its last bind */
};
static struct target targets[TARGETS];
static UINT target_count;
static SRWLOCK targets_lock = SRWLOCK_INIT;
static volatile LONG frame;

/* The target picked at the last present; its snapshot. */
static void *chosen;
static struct
{
    void *of;
    ID3D12Resource *copy;
    volatile LONG copied;  /* the present during whose frame it was last copied, + 1 */
    UINT recorded;         /* draws recorded with a pipeline in the pass (or up to the draw) copied */
    LONG step_frame, step_count;  /* REBOX_DEBUG_VIEW_AFTER: this frame's draws into it so far */
} snap;
static struct { ID3D12Resource *res; LONG at; } retired[8];
static SRWLOCK snap_lock = SRWLOCK_INIT;

static BOOL read_format(DXGI_FORMAT f, DXGI_FORMAT *view, BOOL *hdr);

static float env_float(const char *name, float fallback)
{
    const char *v = getenv(name);
    return v && *v ? (float)atof(v) : fallback;
}

static BOOL CALLBACK init(INIT_ONCE *once, void *param, void **ctx)
{
    const char *v = getenv("REBOX_DEBUG_VIEW"), *c;
    char *end;

    enabled = v && *v && strcmp(v, "0");
    if (!enabled)
        return TRUE;
    snprintf(select_text, sizeof(select_text), "%s", v);
    if (!strcmp(v, "1") || !strcmp(v, "auto"))
        select_kind = SEL_AUTO;
    else if (!strcmp(v, "depth"))
        select_kind = SEL_DEPTH;
    else if (!strncmp(v, "gbuffer", 7))
        select_kind = SEL_GBUFFER, select_slot = min(atoi(v + 7), 7);
    else if (!strcmp(v, "largest"))
        select_kind = SEL_LARGEST;
    else if (!strncmp(v, "0x", 2) || !strncmp(v, "0X", 2))
        select_kind = SEL_ADDRESS, select_address = _strtoui64(v + 2, NULL, 16);
    else if ((select_w = strtoul(v, &end, 10)) && (*end == 'x' || *end == 'X'))
    {
        select_kind = SEL_SIZE;
        select_h = strtoul(end + 1, &end, 10);
        select_format = *end == 'f' ? strtoul(end + 1, NULL, 10) : 0;
    }
    else
    {
        rebox_log("d3d12_xs: REBOX_DEBUG_VIEW=%s isn't a target this knows; showing auto", v);
        select_kind = SEL_AUTO;
    }
    colour_scale = env_float("REBOX_DEBUG_VIEW_SCALE", 1.0f);
    decades = env_float("REBOX_DEBUG_VIEW_DECADES", 4.0f);
    if (decades <= 0.0f)
        decades = 4.0f;
    c = getenv("REBOX_DEBUG_VIEW_CHANNEL");
    channel = !c ? 0 : *c == 'r' ? 1 : *c == 'g' ? 2 : *c == 'b' ? 3 : *c == 'a' ? 4 : 0;
    depth_forward = (c = getenv("REBOX_DEBUG_VIEW_DEPTH")) && !strcmp(c, "forward");
    snapshots = !((c = getenv("REBOX_DEBUG_VIEW_SNAPSHOT")) && *c == '0');
    after_draws = (c = getenv("REBOX_DEBUG_VIEW_AFTER")) ? max(atoi(c), 0) : 0;
    c = getenv("REBOX_DEBUG_VIEW_TEST");
    test.mode = !c || !*c || !strcmp(c, "0") ? TEST_OFF : !strcmp(c, "copy") && snapshots ? TEST_COPY : TEST_TARGET;
    rebox_log("d3d12_xs: REBOX_DEBUG_VIEW=%s: a render target or depth buffer is shown in place of the presented "
              "plane (%s; depth %s, %.1f decades; colour scale %.2f)%s", v,
              !snapshots ? "read at the present" : after_draws ? "copied after a frame's n-th draw into it"
              : "copied when a pass that drew it ends", depth_forward ? "forward" : "reversed", decades, colour_scale,
              test.mode == TEST_TARGET ? "; self-test: a pattern cleared into the target"
              : test.mode == TEST_COPY ? "; self-test: a pattern cleared into its copy" : "");
    if (snapshots && after_draws)
        rebox_log("d3d12_xs: REBOX_DEBUG_VIEW_AFTER=%ld: copied after draw number %ld into it (counting those "
                  "recorded with a pipeline) in each frame", after_draws, after_draws);
    return TRUE;
}

BOOL debug_view_on(void)
{
    static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
    if (enabled < 0)
        InitOnceExecuteOnce(&once, init, NULL, NULL);
    return enabled;
}

static BOOL is_depth_format(DXGI_FORMAT f)
{
    switch (f)
    {
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: case DXGI_FORMAT_R32G8X24_TYPELESS:
    case DXGI_FORMAT_D24_UNORM_S8_UINT: case DXGI_FORMAT_R24G8_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT: case DXGI_FORMAT_D16_UNORM:
        return TRUE;
    default:
        return FALSE;
    }
}

/* Under targets_lock, exclusive: the table index of res, entered (and referenced) if new; -1 if full. */
static int target_index(ID3D12Resource *res, BOOL depth, BOOL display)
{
    UINT i, free_slot = TARGETS;
    for (i = 0; i < target_count; i++)
    {
        if (targets[i].res == res)
        {
            targets[i].display |= display;
            return (int)i;
        }
        if (!targets[i].res && free_slot == TARGETS)
            free_slot = i;
    }
    if (free_slot == TARGETS)
    {
        if (target_count == TARGETS)
            return -1;
        free_slot = target_count++;
    }
    memset(&targets[free_slot], 0, sizeof(targets[free_slot]));
    ID3D12Resource_AddRef(res);
    targets[free_slot].res = res;
    targets[free_slot].desc = ID3D12Resource_GetDesc(res);
    targets[free_slot].depth = depth || is_depth_format(targets[free_slot].desc.Format);
    targets[free_slot].display = display;
    targets[free_slot].seen = frame;
    {
        UINT uploads;
        title_texture_info(res, &targets[free_slot].address, &uploads);
    }
    return (int)free_slot;
}

/* --------------------------------------------------------------- snapshot */

/* What a target is used as around a copy or the test pattern. */
enum use { USE_COPY_SOURCE, USE_WRITE };

/* The barrier from the state (kind 1) or layout (kind 2) a resource is tracked in to a use, or back
 * (forward FALSE). USE_WRITE is render target or depth write. */
static void use_barrier(ID3D12GraphicsCommandList *list, ID3D12Resource *res, UINT kind, UINT from, BOOL depth,
                        enum use use, BOOL forward)
{
    if (kind == 2) /* an enhanced-barrier layout */
    {
        UINT layout = use == USE_COPY_SOURCE ? D3D12_BARRIER_LAYOUT_COPY_SOURCE
                : depth ? D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE : D3D12_BARRIER_LAYOUT_RENDER_TARGET;
        UINT sync = use == USE_COPY_SOURCE ? D3D12_BARRIER_SYNC_COPY
                : depth ? D3D12_BARRIER_SYNC_DEPTH_STENCIL : D3D12_BARRIER_SYNC_RENDER_TARGET;
        UINT access = use == USE_COPY_SOURCE ? D3D12_BARRIER_ACCESS_COPY_SOURCE
                : depth ? D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE : D3D12_BARRIER_ACCESS_RENDER_TARGET;
        D3D12_TEXTURE_BARRIER t = {
            .SyncBefore = forward ? D3D12_BARRIER_SYNC_ALL : sync,
            .SyncAfter = forward ? sync : D3D12_BARRIER_SYNC_ALL,
            .AccessBefore = forward ? D3D12_BARRIER_ACCESS_COMMON : access,
            .AccessAfter = forward ? access : D3D12_BARRIER_ACCESS_COMMON,
            .LayoutBefore = forward ? from : layout,
            .LayoutAfter = forward ? layout : from,
            .pResource = res,
            .Subresources = { 0xffffffff },
        };
        D3D12_BARRIER_GROUP g = { .Type = D3D12_BARRIER_TYPE_TEXTURE, .NumBarriers = 1, .pTextureBarriers = &t };
        if (from != layout)
            ID3D12GraphicsCommandList7_Barrier((ID3D12GraphicsCommandList7 *)list, 1, &g);
    }
    else
    {
        UINT state = use == USE_COPY_SOURCE ? D3D12_RESOURCE_STATE_COPY_SOURCE
                : depth ? D3D12_RESOURCE_STATE_DEPTH_WRITE : D3D12_RESOURCE_STATE_RENDER_TARGET;
        D3D12_RESOURCE_BARRIER b = { .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION };
        if (from == state)
            return;
        b.Transition.pResource = res;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b.Transition.StateBefore = forward ? from : state;
        b.Transition.StateAfter = forward ? state : from;
        ID3D12GraphicsCommandList_ResourceBarrier(list, 1, &b);
    }
}

static void plain_barrier(ID3D12GraphicsCommandList *list, ID3D12Resource *res, D3D12_RESOURCE_STATES before,
                          D3D12_RESOURCE_STATES after)
{
    D3D12_RESOURCE_BARRIER b = { .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION };
    b.Transition.pResource = res;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    ID3D12GraphicsCommandList_ResourceBarrier(list, 1, &b);
}

/* The format a render target or depth view writes a target of format f through (0: none this can). */
static DXGI_FORMAT write_format(DXGI_FORMAT f, BOOL depth)
{
    DXGI_FORMAT view;
    BOOL hdr;

    if (depth)
        switch (f)
        {
        case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: case DXGI_FORMAT_R32G8X24_TYPELESS:
            return DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
        case DXGI_FORMAT_D24_UNORM_S8_UINT: case DXGI_FORMAT_R24G8_TYPELESS: return DXGI_FORMAT_D24_UNORM_S8_UINT;
        case DXGI_FORMAT_D32_FLOAT: case DXGI_FORMAT_R32_TYPELESS: return DXGI_FORMAT_D32_FLOAT;
        case DXGI_FORMAT_D16_UNORM: case DXGI_FORMAT_R16_TYPELESS: return DXGI_FORMAT_D16_UNORM;
        default: return DXGI_FORMAT_UNKNOWN;
        }
    if (!read_format(f, &view, &hdr) || view == DXGI_FORMAT_R9G9B9E5_SHAREDEXP)
        return DXGI_FORMAT_UNKNOWN;
    return view;
}

/* REBOX_DEBUG_VIEW_TEST: the pattern (see the top) cleared into res, a render target or depth buffer
 * in the state or layout (kind) from, left as it was found. Through clears, which go where draws
 * go (render target or depth writes, compression and all). */
static BOOL test_pattern(ID3D12GraphicsCommandList *list, ID3D12Resource *res, UINT kind, UINT from)
{
    static ID3D12DescriptorHeap *rtv_heap, *dsv_heap;
    ID3D12Device *device = rebox_shared_device();
    D3D12_RESOURCE_DESC d = ID3D12Resource_GetDesc(res);
    BOOL depth = !!(d.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
    DXGI_FORMAT format = write_format(d.Format, depth);
    UINT cell_w = max((UINT)(d.Width / 8), 1u), cell_h = max(d.Height / 8, 1u);
    D3D12_CPU_DESCRIPTOR_HANDLE handle;
    HRESULT hr;

    if (!device || !format || d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || d.SampleDesc.Count != 1
            || !(d.Flags & (D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)))
    {
        rebox_log("d3d12_xs: REBOX_DEBUG_VIEW_TEST: no pattern for a %llux%u target of format %u, flags %#x",
                  (unsigned long long)d.Width, d.Height, d.Format, (UINT)d.Flags);
        return FALSE;
    }
    if (!(depth ? dsv_heap : rtv_heap))
    {
        D3D12_DESCRIPTOR_HEAP_DESC hd = { depth ? D3D12_DESCRIPTOR_HEAP_TYPE_DSV : D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1,
                                          D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0 };
        if (FAILED(hr = ID3D12Device_CreateDescriptorHeap(device, &hd, &IID_ID3D12DescriptorHeap,
                                                          (void **)(depth ? &dsv_heap : &rtv_heap))))
        {
            rebox_log("d3d12_xs: REBOX_DEBUG_VIEW_TEST: no descriptor heap (%#lx)", hr);
            return FALSE;
        }
    }
    handle = ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(depth ? dsv_heap : rtv_heap);
    if (depth)
    {
        D3D12_DEPTH_STENCIL_VIEW_DESC v = { .Format = format };
        if (d.DepthOrArraySize > 1)
            v.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY, v.Texture2DArray.ArraySize = 1;
        else
            v.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
        ID3D12Device_CreateDepthStencilView(device, res, &v, handle);
    }
    else
    {
        D3D12_RENDER_TARGET_VIEW_DESC v = { .Format = format };
        if (d.DepthOrArraySize > 1)
            v.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY, v.Texture2DArray.ArraySize = 1;
        else
            v.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        ID3D12Device_CreateRenderTargetView(device, res, &v, handle);
    }

    use_barrier(list, res, kind, from, depth, USE_WRITE, TRUE);
    {
        static const FLOAT base[4] = { 0.5f, 0.0f, 0.5f, 1.0f };
        if (depth)
            ID3D12GraphicsCommandList_ClearDepthStencilView(list, handle, D3D12_CLEAR_FLAG_DEPTH, 0.0f, 0, 0, NULL);
        else
            ID3D12GraphicsCommandList_ClearRenderTargetView(list, handle, base, 0, NULL);
    }
    for (UINT y = 0; y < 8; y++)
        for (UINT x = (y & 1) ^ 1; x < 8; x += 2)
        {
            D3D12_RECT r = { (LONG)(x * cell_w), (LONG)(y * cell_h), (LONG)min((x + 1) * cell_w, (UINT)d.Width),
                             (LONG)min((y + 1) * cell_h, d.Height) };
            if (r.right <= r.left || r.bottom <= r.top)
                continue;
            if (depth)
                ID3D12GraphicsCommandList_ClearDepthStencilView(list, handle, D3D12_CLEAR_FLAG_DEPTH,
                        powf(10.0f, -(float)(x + y) / 4.0f), 0, 1, &r);
            else
            {
                FLOAT c[4] = { (x + 1) / 8.0f, (y + 1) / 8.0f, 1.0f, 1.0f };
                ID3D12GraphicsCommandList_ClearRenderTargetView(list, handle, c, 1, &r);
            }
        }
    use_barrier(list, res, kind, from, depth, USE_WRITE, FALSE);
    rebox_log("d3d12_xs: REBOX_DEBUG_VIEW_TEST: the pattern is cleared into %s %llux%u format %u (written as %u), "
              "in frame %ld; the frame that shows it is dumped", res == snap.copy ? "the copy of the" : "the title's",
              (unsigned long long)d.Width, d.Height, d.Format, format, frame);
    return TRUE;
}

/* Whether a draw was recorded with a pipeline (DRAW_FATE_*). */
static BOOL fate_recorded(UINT fate)
{
    return fate == DRAW_FATE_RECORDED || fate == DRAW_FATE_STANDIN || fate == DRAW_FATE_COMPILED
            || fate == DRAW_FATE_PLAIN;
}

/* Records a copy of the chosen target into its snapshot (made on first use): at the end of a pass
 * that drew it, or after a frame's n-th draw into it (REBOX_DEBUG_VIEW_AFTER). recorded: the draws
 * recorded with a pipeline the copy holds. writing: the target is bound (mid-pass), so it is in a
 * render target or depth write state whatever the last barrier tracked says. */
static void snapshot_copy(ID3D12GraphicsCommandList *list, void *target, UINT recorded, BOOL writing)
{
    ID3D12Resource *copy = NULL, *res = target;
    D3D12_RESOURCE_DESC d = ID3D12Resource_GetDesc(res);
    BOOL depth = !!(d.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL), test_now;
    UINT kind = 1, from = D3D12_RESOURCE_STATE_COMMON;

    AcquireSRWLockExclusive(&snap_lock);
    if (test.state == 1 && test.written == frame && snap.of == target)
    {
        /* The pattern's frame: no later copy covers it before it is shown. */
        ReleaseSRWLockExclusive(&snap_lock);
        return;
    }
    if (snap.of != target || !snap.copy)
    {
        D3D12_HEAP_PROPERTIES heap = { D3D12_HEAP_TYPE_DEFAULT };
        ID3D12Resource *made = NULL;
        HRESULT hr;

        if (snap.copy)
        {
            /* Lists recorded with it may still be in flight: released some presents later. */
            UINT oldest = 0;
            for (UINT i = 1; i < ARRAYSIZE(retired); i++)
                if (!retired[i].res || (retired[oldest].res && retired[i].at < retired[oldest].at))
                    oldest = i;
            if (retired[oldest].res)
                ID3D12Resource_Release(retired[oldest].res);
            retired[oldest].res = snap.copy;
            retired[oldest].at = frame;
        }
        /* The same format and layout, so CopyResource always applies (a depth buffer's copy is one
         * too, stencil plane and all); only the flags that would keep it from being read go (a render
         * target's stays for REBOX_DEBUG_VIEW_TEST=copy, which clears its pattern into it). */
        d.Flags &= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL
                | (test.mode == TEST_COPY ? D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET : 0);
        d.Alignment = 0;
        d.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        hr = ID3D12Device_CreateCommittedResource(rebox_shared_device(), &heap, D3D12_HEAP_FLAG_NONE, &d,
                D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource, (void **)&made);
        snap.copy = SUCCEEDED(hr) ? made : NULL;
        snap.of = target;
        snap.copied = 0;
        snap.recorded = 0;
        if (FAILED(hr))
            rebox_log("d3d12_xs: REBOX_DEBUG_VIEW: no copy of a %llux%u format %u target (%#lx); it is read at the "
                      "present instead", (unsigned long long)d.Width, d.Height, d.Format, hr);
    }
    copy = snap.copy;
    test_now = test.mode && !test.state;
    if (test_now)
        test.state = 1, test.written = frame;
    if (copy)
        snap.recorded = recorded;
    ReleaseSRWLockExclusive(&snap_lock);
    if (!copy)
        return;
    present_tracked(res, &kind, &from);
    if (writing)
    {
        /* Bound and drawn into: what the tracked barrier says unless it isn't a writable state. */
        BOOL bindable = kind == 2 ? from == D3D12_BARRIER_LAYOUT_RENDER_TARGET
                || from == D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE || from == D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ
                : from == D3D12_RESOURCE_STATE_RENDER_TARGET || (from & (D3D12_RESOURCE_STATE_DEPTH_WRITE
                | D3D12_RESOURCE_STATE_DEPTH_READ));
        if (!bindable)
        {
            if (kind == 2)
                from = depth ? D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE : D3D12_BARRIER_LAYOUT_RENDER_TARGET;
            else
                from = depth ? D3D12_RESOURCE_STATE_DEPTH_WRITE : D3D12_RESOURCE_STATE_RENDER_TARGET;
        }
    }
    if (test_now && test.mode == TEST_TARGET)
        test_pattern(list, res, kind, from);
    use_barrier(list, res, kind, from, depth, USE_COPY_SOURCE, TRUE);
    plain_barrier(list, copy, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12GraphicsCommandList_CopyResource(list, copy, res);
    plain_barrier(list, copy, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    use_barrier(list, res, kind, from, depth, USE_COPY_SOURCE, FALSE);
    if (test_now && test.mode == TEST_COPY)
        test_pattern(list, copy, 1, D3D12_RESOURCE_STATE_COMMON);
    InterlockedExchange(&snap.copied, frame + 1);
}

/* The end of a pass that had the chosen target bound: copied, unless its draws into it were all left
 * out (stages compiling: the copy would show the clear), or it drew nothing into it and the copy
 * already holds draws. With REBOX_DEBUG_VIEW_TEST and SNAPSHOT=0, the pattern goes into the target
 * here, read at the present. */
static void pass_end(ID3D12GraphicsCommandList *list, struct debug_view_list *l)
{
    void *picked = chosen;
    BOOL take;

    if (!l->snapshot_res || l->snapshot_res != picked)
        return;
    if (!snapshots)
    {
        if (test.mode && !InterlockedCompareExchange(&test.state, 1, 0))
        {
            UINT kind = 1, from = D3D12_RESOURCE_STATE_COMMON;
            test.written = frame;
            present_tracked(picked, &kind, &from);
            test_pattern(list, picked, kind, from);
        }
        return;
    }
    if (after_draws)
        return;
    AcquireSRWLockShared(&snap_lock);
    take = l->pass_recorded || (!l->pass_draws && (snap.of != picked || !snap.copied || !snap.recorded));
    ReleaseSRWLockShared(&snap_lock);
    if (take || (test.mode && !test.state))
        snapshot_copy(list, picked, l->pass_recorded, FALSE);
    else
    {
        static LONG logged;
        if (l->pass_draws && InterlockedIncrement(&logged) <= 8)
            rebox_log("d3d12_xs: REBOX_DEBUG_VIEW: a pass's %u draws into the target shown were all left out (frame "
                      "%ld); not copied, the last copy with draws stays", l->pass_draws, frame);
    }
}

/* ------------------------------------------------------------ recording */

void debug_view_bind(ID3D12GraphicsCommandList *list, struct debug_view_list *l, ID3D12Resource *const *rts,
                     UINT count, ID3D12Resource *depth, BOOL display)
{
    int indices[8], d = -1;
    UINT n = min(count, 8u);
    void *picked = chosen, *bound_picked = NULL;

    for (UINT i = 0; picked && i < n; i++)
        if (rts[i] == picked)
            bound_picked = picked;
    if (picked && depth == picked)
        bound_picked = picked;
    /* A rebind that keeps it bound continues the pass (titles rebind the same targets between
     * draws); else the pass that had it ends. */
    if (!bound_picked || l->snapshot_res != bound_picked)
    {
        pass_end(list, l);
        l->pass_draws = l->pass_recorded = 0;
    }
    AcquireSRWLockExclusive(&targets_lock);
    for (UINT i = 0; i < n; i++)
        indices[i] = rts[i] ? target_index(rts[i], FALSE, display) : -1;
    if (depth)
        d = target_index(depth, TRUE, display);
    if (n && indices[0] >= 0)
    {
        memset(targets[indices[0]].partners, 0, sizeof(targets[indices[0]].partners));
        for (UINT i = 1; i < n; i++)
            targets[indices[0]].partners[i] = indices[i] + 1;
        InterlockedIncrement(&targets[indices[0]].binds);
    }
    if (d >= 0)
        InterlockedIncrement(&targets[d].binds);
    l->rt = n && indices[0] >= 0 ? indices[0] + 1 : 0;
    l->rt_res = l->rt ? targets[indices[0]].res : NULL;
    l->depth = d + 1;
    l->depth_res = d >= 0 ? targets[d].res : NULL;
    ReleaseSRWLockExclusive(&targets_lock);
    l->snapshot_res = bound_picked;
}

/* ---------------------------------------------------------------- probe */

/* REBOX_DEBUG_VIEW_PROBE=<n> (default 2, 0 off): in n frames (at least 10 s apart) where the target
 * shown had 64 draws or more in the frame before, each draw into it (up to PROBE_SLOTS) gets an
 * occlusion and a pipeline statistics query, and the draw or mesh dispatch arguments of an
 * ExecuteIndirect's first record (past the root arguments the signature sets first; and its count)
 * are copied as they stand at the draw. The results are logged once the GPU has them: whether
 * the draws recorded with a pipeline dispatched any groups, got primitives to the clipper, kept
 * any past it, ran any pixels and passed any samples, which tells where a depth buffer still at its
 * clear value lost them; with each draw's viewport and scissor and the inputs it read as empty
 * (guest buffers and root views in title memory not made GPU memory). */
#define PROBE_SLOTS 512
#define PROBE_STRIDE 128   /* readback bytes a slot: occlusion 8, statistics 88, arguments 20, count 4 */
#define PROBE_STATS 8
#define PROBE_ARGS 96
#define PROBE_ARG_BYTES 20
#define PROBE_COUNT 116

static struct
{
    ID3D12QueryHeap *occlusion, *stats;
    ID3D12Resource *readback;
    volatile BYTE *data;
    int remaining;         /* probes left; -1: not read yet */
    volatile LONG frame;   /* the frame whose draws are probed, or 0 */
    volatile LONG used;    /* slots handed out */
    ULONGLONG last_tick;
    struct
    {
        BYTE recorded, mesh, fate, args_words, has_count;
        /* As the draw was recorded: the list's first viewport and scissor (counts 0: none set
         * since its reset), its graphics root views read as empty (a bit each), and the guest
         * buffers its stages had read as empty. */
        BYTE viewports, scissors;
        D3D12_VIEWPORT viewport;
        RECT scissor;
        UINT root_views_empty, empty_buffers;
        /* The first root view read as empty's address (root index: the lowest bit of
         * root_views_empty), and where the arguments are read from (GPU VA, 0: no ExecuteIndirect)
         * and whether that is in Rebox's GPU memory chunks (title memory made GPU memory). */
        UINT64 empty_view_address, args_va;
        BYTE args_in_chunks;
    } slots[PROBE_SLOTS];
} probe = { .remaining = -1 };

static BOOL probe_setup(void)
{
    ID3D12Device *device = rebox_shared_device();
    D3D12_QUERY_HEAP_DESC query = { D3D12_QUERY_HEAP_TYPE_OCCLUSION, PROBE_SLOTS, 0 };
    D3D12_HEAP_PROPERTIES heap = { D3D12_HEAP_TYPE_READBACK };
    D3D12_RESOURCE_DESC desc = { D3D12_RESOURCE_DIMENSION_BUFFER, 0, PROBE_SLOTS * PROBE_STRIDE, 1, 1, 1,
                                 DXGI_FORMAT_UNKNOWN, { 1, 0 }, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE };
    void *mapped = NULL;
    HRESULT hr;

    if (probe.data)
        return TRUE;
    if (!device)
        return FALSE;
    if (FAILED(hr = ID3D12Device_CreateQueryHeap(device, &query, &IID_ID3D12QueryHeap, (void **)&probe.occlusion)))
        goto fail;
    query.Type = D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS;
    if (FAILED(hr = ID3D12Device_CreateQueryHeap(device, &query, &IID_ID3D12QueryHeap, (void **)&probe.stats)))
        goto fail;
    if (FAILED(hr = ID3D12Device_CreateCommittedResource(device, &heap, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_COPY_DEST, NULL, &IID_ID3D12Resource, (void **)&probe.readback)))
        goto fail;
    if (FAILED(hr = ID3D12Resource_Map(probe.readback, 0, NULL, &mapped)))
        goto fail;
    probe.data = mapped;
    return TRUE;

fail:
    rebox_log("d3d12_xs: REBOX_DEBUG_VIEW_PROBE: setup failed (%#lx); no probe", hr);
    probe.remaining = 0;
    return FALSE;
}

/* Under targets_lock, at a present: probe this frame's draws into the target picked. */
static void probe_arm(LONG now, const struct target *t)
{
    ULONGLONG tick = GetTickCount64();

    if (probe.remaining < 0)
    {
        const char *v = getenv("REBOX_DEBUG_VIEW_PROBE");
        probe.remaining = v && *v ? atoi(v) : 2;
    }
    if (probe.remaining <= 0 || probe.frame || t->draws < 64 || (probe.last_tick && tick - probe.last_tick < 10000)
            || !probe_setup())
        return;
    memset((void *)probe.data, 0xff, PROBE_SLOTS * PROBE_STRIDE);
    memset(probe.slots, 0, sizeof(probe.slots));
    probe.used = 0;
    probe.last_tick = tick;
    probe.remaining--;
    InterlockedExchange(&probe.frame, now);
    rebox_log("d3d12_xs: REBOX_DEBUG_VIEW_PROBE: the draws of frame %ld into the %s %llux%u shown get queries", now,
              t->depth ? "depth buffer" : "target", (unsigned long long)t->desc.Width, t->desc.Height);
}

static UINT64 probe_u64(UINT slot, UINT at)
{
    UINT64 v;
    memcpy(&v, (const BYTE *)probe.data + slot * PROBE_STRIDE + at, sizeof(v));
    return v;
}

static UINT probe_u32(UINT slot, UINT at)
{
    UINT v;
    memcpy(&v, (const BYTE *)probe.data + slot * PROBE_STRIDE + at, sizeof(v));
    return v;
}

/* At a present after the probed frame: the results, once every query recorded has landed (or 8
 * presents on: lists never submitted). */
/* A probed draw's viewport and scissor, as text. */
static void probe_view_text(UINT i, char *out, size_t size)
{
    const D3D12_VIEWPORT *v = &probe.slots[i].viewport;
    const RECT *r = &probe.slots[i].scissor;
    int at;

    if (probe.slots[i].viewports)
        at = snprintf(out, size, "viewport %g,%g %gx%g depth %g-%g", v->TopLeftX, v->TopLeftY, v->Width, v->Height,
                      v->MinDepth, v->MaxDepth);
    else
        at = snprintf(out, size, "no viewport set");
    if (at < 0 || (size_t)at >= size)
        return;
    if (probe.slots[i].scissors)
        snprintf(out + at, size - at, ", scissor %ld,%ld-%ld,%ld", r->left, r->top, r->right, r->bottom);
    else
        snprintf(out + at, size - at, ", no scissor set");
}

/* Whether a probed draw's viewport or scissor covers no pixel; one not set since the list's reset
 * counts (D3D12's default is empty). */
static BOOL probe_view_empty(UINT i)
{
    const D3D12_VIEWPORT *v = &probe.slots[i].viewport;
    const RECT *r = &probe.slots[i].scissor;
    return !probe.slots[i].viewports || !probe.slots[i].scissors || !(v->Width > 0.0f) || !(v->Height > 0.0f)
            || r->right <= r->left || r->bottom <= r->top;
}

static void probe_report(LONG now)
{
    struct sums
    {
        UINT draws, resolved, with_args, with_groups, with_clipped_in, with_clipped_out, with_pixels, with_samples;
        UINT with_empty_view, with_empty_inputs;
        UINT64 groups, clip_in, clip_out, pixels, samples;
    } sums[3] = { { 0 } };  /* mesh draws recorded, other draws recorded, draws left out */
    static const char *const names[3] = { "mesh draws recorded", "other draws recorded", "draws left out" };
    UINT used = min((UINT)probe.used, (UINT)PROBE_SLOTS), pending = 0, detailed = 0;

    if (!probe.frame || now <= probe.frame)
        return;
    for (UINT i = 0; i < used; i++)
        pending += probe.slots[i].recorded && probe_u64(i, 0) == ~0ull;
    if (pending && now - probe.frame < 8)
        return;
    rebox_log("d3d12_xs: REBOX_DEBUG_VIEW_PROBE: frame %ld's %u draws into the target shown%s (%u of them not "
              "resolved: their lists never ran, or not yet):", probe.frame, used,
              (UINT)probe.used > PROBE_SLOTS ? " (the first of more)" : "", pending);
    for (UINT i = 0; i < used; i++)
    {
        const D3D12_QUERY_DATA_PIPELINE_STATISTICS *st;
        D3D12_QUERY_DATA_PIPELINE_STATISTICS stats;
        BOOL recorded = fate_recorded(probe.slots[i].fate);
        struct sums *s = &sums[!recorded ? 2 : probe.slots[i].mesh ? 0 : 1];
        UINT64 samples = probe_u64(i, 0), groups = 0;
        UINT args[5] = { 0 }, count = ~0u;

        s->draws++;
        if (!probe.slots[i].recorded || samples == ~0ull)
            continue;
        memcpy(&stats, (const BYTE *)probe.data + i * PROBE_STRIDE + PROBE_STATS, sizeof(stats));
        st = &stats;
        s->resolved++;
        for (UINT w = 0; w < probe.slots[i].args_words && w < 5; w++)
            args[w] = probe_u32(i, PROBE_ARGS + w * 4);
        if (probe.slots[i].has_count)
            count = probe_u32(i, PROBE_COUNT);
        if (probe.slots[i].args_words >= 3)
        {
            /* A mesh record's group counts; a draw's vertex (index) count and instances. */
            groups = probe.slots[i].mesh ? (UINT64)args[0] * args[1] * args[2] : (UINT64)args[0] * args[1];
            if (count != ~0u && !count)
                groups = 0;
            s->with_args++;
            s->with_groups += groups != 0;
            s->groups += groups;
        }
        s->with_clipped_in += st->CInvocations != 0;
        s->with_clipped_out += st->CPrimitives != 0;
        s->with_pixels += st->PSInvocations != 0;
        s->with_samples += samples != 0;
        s->clip_in += st->CInvocations;
        s->clip_out += st->CPrimitives;
        s->pixels += st->PSInvocations;
        s->samples += samples;
        s->with_empty_view += probe_view_empty(i);
        s->with_empty_inputs += probe.slots[i].empty_buffers || probe.slots[i].root_views_empty;
        if (recorded && probe.slots[i].mesh && detailed < 12)
        {
            char view[160], where[160];
            int n = 0;
            detailed++;
            probe_view_text(i, view, sizeof(view));
            where[0] = 0;
            if (probe.slots[i].args_va)
                n += snprintf(where + n, sizeof(where) - n, "; arguments at %#llx (%s)",
                              (unsigned long long)probe.slots[i].args_va,
                              probe.slots[i].args_in_chunks ? "Rebox GPU memory" : "vkd3d-proton's own memory");
            if (probe.slots[i].root_views_empty && n < (int)sizeof(where))
            {
                UINT b = 0;
                while (!(probe.slots[i].root_views_empty & (1u << b)))
                    b++;
                snprintf(where + n, sizeof(where) - n, "; root view %u at %#llx", b,
                         (unsigned long long)probe.slots[i].empty_view_address);
            }
            rebox_log("d3d12_xs:   draw %u (%s): arguments %u %u %u%s%s, clipper %llu in %llu out, pixels %llu, samples "
                      "%llu; %s; read as empty: %u guest buffers, root views %#x%s", i,
                      console_draw_fate_name(probe.slots[i].fate), args[0], args[1], args[2],
                      probe.slots[i].args_words ? "" : " (not read)",
                      probe.slots[i].has_count ? (count ? ", counted" : ", count 0") : "",
                      (unsigned long long)st->CInvocations, (unsigned long long)st->CPrimitives,
                      (unsigned long long)st->PSInvocations, (unsigned long long)samples, view,
                      probe.slots[i].empty_buffers, probe.slots[i].root_views_empty, where);
        }
    }
    for (UINT k = 0; k < 3; k++)
        if (sums[k].draws)
            rebox_log("d3d12_xs:   %s: %u, %u resolved; arguments read for %u, %u of them with work (%llu groups or "
                      "vertices x instances); primitives into the clipper in %u (%llu), out of it in %u (%llu); pixels "
                      "in %u (%llu); samples passed in %u (%llu); an empty or unset viewport or scissor in %u; inputs "
                      "read as empty (guest buffers, root views) in %u", names[k], sums[k].draws, sums[k].resolved,
                      sums[k].with_args, sums[k].with_groups, (unsigned long long)sums[k].groups,
                      sums[k].with_clipped_in, (unsigned long long)sums[k].clip_in, sums[k].with_clipped_out,
                      (unsigned long long)sums[k].clip_out, sums[k].with_pixels, (unsigned long long)sums[k].pixels,
                      sums[k].with_samples, (unsigned long long)sums[k].samples, sums[k].with_empty_view,
                      sums[k].with_empty_inputs);
    {
        const struct sums *m = &sums[0];
        const char *verdict;
        if (!m->resolved)
            verdict = "no recorded mesh draw was resolved";
        else if (m->with_samples)
            verdict = "mesh draws passed samples: depth was reached (unless depth writes are off), so a view still at "
                      "the clear value points at the read side (a later clear, another resource, HTILE)";
        else if (m->with_clipped_out && m->with_empty_view == m->resolved)
            verdict = "primitives left the clipper but every mesh draw's viewport or scissor was empty or unset";
        else if (m->with_clipped_out && m->with_empty_inputs)
            verdict = "primitives left the clipper but no sample passed, and mesh draws read inputs as empty (title "
                      "memory not made GPU memory): positions all alike (zero area), else the depth or stencil test";
        else if (m->with_clipped_out)
            verdict = "primitives left the clipper but no sample passed: zero-area primitives (positions all alike), "
                      "or the depth or stencil test, depth bounds, scissor or viewport";
        else if (m->with_clipped_in)
            verdict = "primitives reached the clipper and all were clipped or culled: positions (w 0, NaN, outside) "
                      "or cull flags";
        else if (m->with_args && !m->with_groups)
            verdict = "every argument record read was empty at the draw: the compute pass that fills them (meshlet "
                      "culling) didn't run or wrote nothing";
        else if (m->with_groups)
            verdict = "argument records had groups but no primitive reached the clipper: vkd3d-proton read other "
                      "arguments (REBOX_INDIRECT_BARRIER) or the mesh stage output no primitives (vertex or primitive "
                      "counts, every primitive culled)";
        else
            verdict = "no primitive reached the clipper";
        rebox_log("d3d12_xs:   so: %s", verdict);
    }
    InterlockedExchange(&probe.frame, 0);
}

void debug_view_draw_begin(ID3D12GraphicsCommandList *list, struct debug_view_list *l, ID3D12Resource *args,
                           UINT64 args_offset, ID3D12Resource *count, UINT64 count_offset)
{
    void *picked = chosen;
    LONG slot;

    l->probe = 0;
    if (!probe.frame || probe.frame != frame || !picked || (l->depth_res != picked && l->rt_res != picked))
        return;
    if ((slot = InterlockedIncrement(&probe.used) - 1) >= PROBE_SLOTS)
        return;
    if (args || count)
    {
        /* Every earlier write before the copies (not taken as an indirect barrier by vkd3d-proton,
         * so the draw's own reading stays where it was). */
        D3D12_GLOBAL_BARRIER g = { .SyncBefore = D3D12_BARRIER_SYNC_ALL, .SyncAfter = D3D12_BARRIER_SYNC_COPY,
                                   .AccessBefore = D3D12_BARRIER_ACCESS_COMMON,
                                   .AccessAfter = D3D12_BARRIER_ACCESS_COPY_SOURCE };
        D3D12_BARRIER_GROUP group = { .Type = D3D12_BARRIER_TYPE_GLOBAL, .NumBarriers = 1, .pGlobalBarriers = &g };
        ID3D12GraphicsCommandList7_Barrier((ID3D12GraphicsCommandList7 *)list, 1, &group);
    }
    if (args)
    {
        D3D12_RESOURCE_DESC d = ID3D12Resource_GetDesc(args);
        if (d.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER && args_offset < d.Width)
        {
            UINT64 n = min((UINT64)PROBE_ARG_BYTES, d.Width - args_offset) & ~3ull;
            ID3D12GraphicsCommandList_CopyBufferRegion(list, probe.readback, (UINT64)slot * PROBE_STRIDE + PROBE_ARGS,
                                                       args, args_offset, n);
            probe.slots[slot].args_words = (BYTE)(n / 4);
        }
        else
        {
            static LONG logged;
            if (InterlockedIncrement(&logged) <= 4)
                rebox_log("d3d12_xs: REBOX_DEBUG_VIEW_PROBE: an ExecuteIndirect's arguments at %#llx aren't read: "
                          "its argument resource has dimension %u, width %#llx", (unsigned long long)args_offset,
                          (UINT)d.Dimension, (unsigned long long)d.Width);
        }
    }
    if (count)
    {
        D3D12_RESOURCE_DESC d = ID3D12Resource_GetDesc(count);
        if (d.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER && count_offset + 4 <= d.Width)
        {
            ID3D12GraphicsCommandList_CopyBufferRegion(list, probe.readback, (UINT64)slot * PROBE_STRIDE + PROBE_COUNT,
                                                       count, count_offset, 4);
            probe.slots[slot].has_count = 1;
        }
    }
    probe.slots[slot].viewports = (BYTE)min(l->viewports, 255u);
    probe.slots[slot].scissors = (BYTE)min(l->scissors, 255u);
    probe.slots[slot].viewport = l->viewport;
    probe.slots[slot].scissor = l->scissor;
    probe.slots[slot].root_views_empty = l->root_views_empty;
    probe.slots[slot].empty_view_address = 0;
    for (UINT b = 0; b < ARRAYSIZE(l->root_view_address); b++)
        if (l->root_views_empty & (1u << b))
        {
            probe.slots[slot].empty_view_address = l->root_view_address[b];
            break;
        }
    probe.slots[slot].args_va = 0;
    probe.slots[slot].args_in_chunks = 0;
    if (args)
    {
        UINT64 va = ID3D12Resource_GetGPUVirtualAddress(args);
        if (va)
        {
            probe.slots[slot].args_va = va + args_offset;
            probe.slots[slot].args_in_chunks = (BYTE)gpu_memory_holds_gpu(va + args_offset, 12);
        }
    }
    ID3D12GraphicsCommandList_BeginQuery(list, probe.occlusion, D3D12_QUERY_TYPE_OCCLUSION, slot);
    ID3D12GraphicsCommandList_BeginQuery(list, probe.stats, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, slot);
    l->probe = slot + 1;
}

void debug_view_draw(ID3D12GraphicsCommandList *list, struct debug_view_list *l, BOOL mesh, UINT fate)
{
    fate = min(fate, DRAW_FATES - 1u);
    if (l->probe)
    {
        UINT slot = l->probe - 1;
        ID3D12GraphicsCommandList_EndQuery(list, probe.occlusion, D3D12_QUERY_TYPE_OCCLUSION, slot);
        ID3D12GraphicsCommandList_EndQuery(list, probe.stats, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, slot);
        ID3D12GraphicsCommandList_ResolveQueryData(list, probe.occlusion, D3D12_QUERY_TYPE_OCCLUSION, slot, 1,
                                                   probe.readback, (UINT64)slot * PROBE_STRIDE);
        ID3D12GraphicsCommandList_ResolveQueryData(list, probe.stats, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, slot, 1,
                                                   probe.readback, (UINT64)slot * PROBE_STRIDE + PROBE_STATS);
        probe.slots[slot].mesh = (BYTE)mesh;
        probe.slots[slot].fate = (BYTE)fate;
        probe.slots[slot].empty_buffers = console_draw_watch_empty_buffers();
        probe.slots[slot].recorded = 1;
        l->probe = 0;
    }
    if (l->snapshot_res && l->snapshot_res == chosen)
    {
        BOOL recorded = fate_recorded(fate);
        l->pass_draws++;
        l->pass_recorded += recorded;
        if (snapshots && after_draws && recorded)
        {
            /* REBOX_DEBUG_VIEW_AFTER: this frame's n-th draw into it, counted across lists. */
            LONG n;
            AcquireSRWLockExclusive(&snap_lock);
            if (snap.step_frame != frame)
                snap.step_frame = frame, snap.step_count = 0;
            n = ++snap.step_count;
            ReleaseSRWLockExclusive(&snap_lock);
            if (n == after_draws)
                snapshot_copy(list, l->snapshot_res, (UINT)n, TRUE);
        }
    }
    if (l->rt && targets[l->rt - 1].res == l->rt_res)
    {
        InterlockedIncrement(&targets[l->rt - 1].draws);
        InterlockedIncrement(&targets[l->rt - 1].fates[fate]);
        if (mesh)
            InterlockedIncrement(&targets[l->rt - 1].mesh);
    }
    if (l->depth && targets[l->depth - 1].res == l->depth_res)
    {
        InterlockedIncrement(&targets[l->depth - 1].draws);
        InterlockedIncrement(&targets[l->depth - 1].fates[fate]);
        if (mesh)
            InterlockedIncrement(&targets[l->depth - 1].mesh);
    }
}

/* Under targets_lock: what became of this frame's draws into the target shown, when there were
 * enough to matter (a scene, not a loading screen's few) and at most every 10 s: whether any of
 * them reached the GPU, or why not. */
static void log_fates(LONG now, const struct target *t)
{
    static ULONGLONG last;
    ULONGLONG tick = GetTickCount64();
    char line[1024];
    int at = 0;

    if (t->draws < 64 || (last && tick - last < 10000))
        return;
    last = tick;
    for (UINT f = 0; f < DRAW_FATES && at < (int)sizeof(line) - 96; f++)
        if (t->fates[f])
            at += snprintf(line + at, sizeof(line) - at, "%s %ld %s", at ? "," : "", t->fates[f],
                           console_draw_fate_name(f));
    line[at] = 0;
    rebox_log("d3d12_xs: REBOX_DEBUG_VIEW: present %ld: the %ld draws (%ld mesh) into the %s %llux%u shown:%s", now,
              t->draws, t->mesh, t->depth ? "depth buffer" : "target", (unsigned long long)t->desc.Width,
              t->desc.Height, line);
}

void debug_view_close(ID3D12GraphicsCommandList *list, struct debug_view_list *l)
{
    pass_end(list, l);
    memset(l, 0, sizeof(*l));
}

/* --------------------------------------------------------------- present */

static UINT64 area(const struct target *t)
{
    return t->desc.Width * (UINT64)t->desc.Height;
}

/* Under targets_lock: the entry the setting picks from this frame's counts, or -1. */
static int choose(const char **why)
{
    /* The G-buffer target last picked for its mesh draws: kept through frames where no pass drew
     * meshes (its copy, from the last pass that drew it, is shown), rather than flipping to the
     * depth buffer or the plane and starting a new copy. */
    static int sticky = -1;
    static void *sticky_res;
    int best = -1;

    if (select_kind == SEL_AUTO || select_kind == SEL_GBUFFER)
    {
        /* Render target 0 of the pass with the most mesh draws (then draws). */
        for (UINT i = 0; i < target_count; i++)
        {
            const struct target *t = &targets[i];
            if (!t->res || t->depth || t->display || !t->draws)
                continue;
            if (best < 0 || t->mesh > targets[best].mesh
                    || (t->mesh == targets[best].mesh && t->draws > targets[best].draws))
                best = (int)i;
        }
        if (best >= 0 && targets[best].mesh)
            sticky = best, sticky_res = targets[best].res;
        else if (sticky >= 0 && targets[sticky].res == sticky_res && sticky_res)
        {
            *why = "render target 0 of the pass that last drew meshes (none drew any this frame)";
            if (select_kind == SEL_GBUFFER && select_slot)
            {
                int partner = targets[sticky].partners[select_slot] - 1;
                *why = "a render target of the pass that last drew meshes (none drew any this frame)";
                return partner >= 0 && targets[partner].res ? partner : -1;
            }
            return sticky;
        }
        else
            sticky = -1, sticky_res = NULL;
        if (best >= 0 && (select_kind == SEL_GBUFFER || targets[best].mesh))
        {
            *why = targets[best].mesh ? "render target 0 of the pass with the most mesh draws"
                                      : "render target 0 of the pass with the most draws (none drew meshes)";
            if (select_kind == SEL_GBUFFER && select_slot)
            {
                int partner = targets[best].partners[select_slot] - 1;
                *why = "a render target of the pass with the most mesh draws";
                return partner >= 0 && targets[partner].res ? partner : -1;
            }
            return best;
        }
        if (select_kind == SEL_GBUFFER)
            return -1;
        best = -1;
    }
    if (select_kind == SEL_AUTO || select_kind == SEL_DEPTH)
    {
        for (UINT i = 0; i < target_count; i++)
        {
            const struct target *t = &targets[i];
            if (!t->res || !t->depth || t->display || !t->draws || t->desc.Width == t->desc.Height)
                continue;
            if (best < 0 || t->mesh * 4 + t->draws > targets[best].mesh * 4 + targets[best].draws)
                best = (int)i;
        }
        *why = "the depth buffer with the most draws";
        if (best >= 0 || select_kind == SEL_DEPTH)
            return best;
    }
    if (select_kind == SEL_AUTO || select_kind == SEL_LARGEST)
    {
        for (UINT i = 0; i < target_count; i++)
        {
            const struct target *t = &targets[i];
            if (!t->res || t->depth || t->display || !t->draws)
                continue;
            if (best < 0 || area(t) > area(&targets[best])
                    || (area(t) == area(&targets[best]) && t->draws > targets[best].draws))
                best = (int)i;
        }
        *why = "the largest render target drawn";
        return best;
    }
    if (select_kind == SEL_SIZE)
    {
        for (UINT i = 0; i < target_count; i++)
        {
            const struct target *t = &targets[i];
            if (!t->res || t->desc.Width != select_w || t->desc.Height != select_h
                    || (select_format && t->desc.Format != select_format))
                continue;
            if (best < 0 || t->mesh * 4 + t->draws > targets[best].mesh * 4 + targets[best].draws)
                best = (int)i;
        }
        *why = "the most-drawn target of that size";
        return best;
    }
    if (select_kind == SEL_ADDRESS)
    {
        for (UINT i = 0; i < target_count; i++)
            if (targets[i].res && targets[i].address == select_address)
                best = (int)i;
        *why = "the target at that address";
        return best;
    }
    return -1;
}

static void log_candidates(LONG now)
{
    BOOL taken[TARGETS] = { 0 };
    UINT drawn = 0;

    for (UINT i = 0; i < target_count; i++)
        drawn += targets[i].res && targets[i].draws;
    rebox_log("d3d12_xs: REBOX_DEBUG_VIEW: present %ld: %u targets drawn; the most drawn:", now, drawn);
    for (UINT k = 0; k < 8; k++)
    {
        int best = -1;
        const struct target *t;
        for (UINT i = 0; i < target_count; i++)
            if (targets[i].res && targets[i].draws && !taken[i] && (best < 0 || targets[i].draws > targets[best].draws))
                best = (int)i;
        if (best < 0)
            break;
        taken[best] = TRUE;
        t = &targets[best];
        rebox_log("d3d12_xs:   %s %llux%u format %u at %#llx: %ld draws (%ld mesh), %ld binds%s",
                  t->depth ? "depth" : "colour", (unsigned long long)t->desc.Width, t->desc.Height, t->desc.Format,
                  (unsigned long long)t->address, t->draws, t->mesh, t->binds, t->display ? ", with the display" : "");
    }
}

BOOL debug_view_pick(struct debug_view_pick *pick)
{
    static void *last;
    static LONG nothing_logged;
    const char *why = "";
    LONG now;
    int i;

    memset(pick, 0, sizeof(*pick));
    if (!debug_view_on())
        return FALSE;
    AcquireSRWLockExclusive(&targets_lock);
    now = InterlockedIncrement(&frame);
    i = choose(&why);
    probe_report(now);
    if (!(now % 60))
        log_candidates(now);
    if (i >= 0)
    {
        struct target *t = &targets[i];
        if (t->res != last)
            rebox_log("d3d12_xs: REBOX_DEBUG_VIEW=%s at present %ld: showing %s, %s %llux%u format %u at %#llx "
                      "(%ld draws, %ld of them mesh, in %ld binds)", select_text, now, why, t->depth ? "depth" : "colour",
                      (unsigned long long)t->desc.Width, t->desc.Height, t->desc.Format, (unsigned long long)t->address,
                      t->draws, t->mesh, t->binds);
        last = t->res;
        log_fates(now, t);
        probe_arm(now, t);
        pick->desc = t->desc;
        pick->depth = t->depth;
        pick->res = t->res;
    }
    else if (select_kind == SEL_ADDRESS && (pick->res = title_texture_at(select_address)))
    {
        pick->desc = ID3D12Resource_GetDesc(pick->res);
        pick->depth = is_depth_format(pick->desc.Format);
        if (pick->res != last)
            rebox_log("d3d12_xs: REBOX_DEBUG_VIEW=%s at present %ld: showing the texture placed there, %llux%u "
                      "format %u (never bound as a target)", select_text, now, (unsigned long long)pick->desc.Width,
                      pick->desc.Height, pick->desc.Format);
        last = pick->res;
    }
    else if (!(InterlockedIncrement(&nothing_logged) % 60) || nothing_logged == 1)
        rebox_log("d3d12_xs: REBOX_DEBUG_VIEW=%s at present %ld: nothing to show yet (%u targets known); the plane "
                  "is shown", select_text, now, target_count);
    chosen = pick->res;
    /* A new frame's counts; targets unbound for a while are let go. */
    for (UINT k = 0; k < target_count; k++)
    {
        struct target *t = &targets[k];
        if (!t->res)
            continue;
        if (t->draws || t->binds)
            t->seen = now;
        t->draws = t->mesh = t->binds = 0;
        memset((void *)t->fates, 0, sizeof(t->fates));
        if (now - t->seen > EVICT_PRESENTS && t->res != pick->res && t->res != snap.of)
        {
            ID3D12Resource_Release(t->res);
            t->res = NULL;
        }
    }
    if (pick->res)
        ID3D12Resource_AddRef(pick->res);
    ReleaseSRWLockExclusive(&targets_lock);
    if (!pick->res)
        return FALSE;

    /* Its last copy, however old: passes whose draws were all left out aren't copied (pass_end). */
    AcquireSRWLockExclusive(&snap_lock);
    for (UINT k = 0; k < ARRAYSIZE(retired); k++)
        if (retired[k].res && now - retired[k].at > EVICT_PRESENTS)
        {
            ID3D12Resource_Release(retired[k].res);
            retired[k].res = NULL;
        }
    if (snapshots && snap.of == pick->res && snap.copy && snap.copied)
    {
        LONG at = snap.copied - 1;
        snprintf(pick->note, sizeof(pick->note), "the copy of the %s %llux%u format %u made in frame %ld (%ld "
                 "presents ago), %u draws recorded with a pipeline in its pass%s", pick->depth ? "depth buffer" : "target",
                 (unsigned long long)pick->desc.Width, pick->desc.Height, pick->desc.Format, at, now - 1 - at,
                 snap.recorded, after_draws ? " (up to the draw copied after)" : "");
        ID3D12Resource_Release(pick->res);
        pick->res = snap.copy;
        ID3D12Resource_AddRef(pick->res);
        pick->snapshot = TRUE;
    }
    else
        snprintf(pick->note, sizeof(pick->note), "the %s %llux%u format %u itself, read at the present%s",
                 pick->depth ? "depth buffer" : "target", (unsigned long long)pick->desc.Width, pick->desc.Height,
                 pick->desc.Format, snapshots ? " (no copy of it made yet)" : "");
    /* REBOX_DEBUG_VIEW_TEST: the first present after the pattern was written shows it. */
    if (test.state == 1 && now > test.written && (pick->snapshot || !snapshots))
    {
        test.state = 2;
        pick->dump = TRUE;
        rebox_log("d3d12_xs: REBOX_DEBUG_VIEW_TEST: present %ld shows the pattern written in frame %ld (%s); its frame "
                  "is dumped: an 8x8 grid there proves the copy, the view and the dump", now, test.written, pick->note);
    }
    ReleaseSRWLockExclusive(&snap_lock);
    return TRUE;
}

/* ------------------------------------------------------------- the view */

static ID3D12RootSignature *root_signature;
static ID3D12PipelineState *pipeline;
static ID3D12DescriptorHeap *heap;
static UINT increment;
static LONG next_slot;

#define RING 64

static BOOL CALLBACK init_pipeline(INIT_ONCE *once, void *param, void **ctx)
{
    ID3D12Device *device = rebox_shared_device();
    D3D12_DESCRIPTOR_RANGE ranges[] = {
        { D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0 },
        { D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 1 },
    };
    D3D12_ROOT_PARAMETER params[2] = {
        { .ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE, .DescriptorTable = { 2, ranges } },
        { .ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS, .Constants = { 0, 0, 8 } },
    };
    D3D12_ROOT_SIGNATURE_DESC rs_desc = { 2, params, 0, NULL, D3D12_ROOT_SIGNATURE_FLAG_NONE };
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc = { D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, RING * 2,
                                             D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0 };
    D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc = { 0 };
    PFN_D3D12_SERIALIZE_ROOT_SIGNATURE serialize;
    ID3D10Blob *blob = NULL;
    HMODULE d3d12;
    HRESULT hr = E_FAIL;

    if (!device || !(d3d12 = GetModuleHandleA("d3d12.dll"))
        || !(serialize = (PFN_D3D12_SERIALIZE_ROOT_SIGNATURE)(void *)GetProcAddress(d3d12, "D3D12SerializeRootSignature")))
        goto fail;
    if (FAILED(hr = serialize(&rs_desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, NULL))
        || FAILED(hr = ID3D12Device_CreateRootSignature(device, 0, ID3D10Blob_GetBufferPointer(blob),
                ID3D10Blob_GetBufferSize(blob), &IID_ID3D12RootSignature, (void **)&root_signature)))
        goto fail;
    pso_desc.pRootSignature = root_signature;
    pso_desc.CS.pShaderBytecode = debug_view_cs;
    pso_desc.CS.BytecodeLength = sizeof(debug_view_cs);
    if (FAILED(hr = ID3D12Device_CreateComputePipelineState(device, &pso_desc, &IID_ID3D12PipelineState,
                                                            (void **)&pipeline))
        || FAILED(hr = ID3D12Device_CreateDescriptorHeap(device, &heap_desc, &IID_ID3D12DescriptorHeap, (void **)&heap)))
        goto fail;
    increment = ID3D12Device_GetDescriptorHandleIncrementSize(device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    ID3D10Blob_Release(blob);
    return TRUE;

fail:
    rebox_log("d3d12_xs: REBOX_DEBUG_VIEW: setup failed: %#lx", hr);
    if (blob)
        ID3D10Blob_Release(blob);
    pipeline = NULL;
    return TRUE;
}

/* The view format a shader reads a target's first plane through, and whether it holds floats
 * (FALSE: integer formats, which this can't show). */
static BOOL read_format(DXGI_FORMAT f, DXGI_FORMAT *view, BOOL *hdr)
{
    *hdr = FALSE;
    switch (f)
    {
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: case DXGI_FORMAT_R32G8X24_TYPELESS:
        *view = DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS; return TRUE;
    case DXGI_FORMAT_D24_UNORM_S8_UINT: case DXGI_FORMAT_R24G8_TYPELESS:
        *view = DXGI_FORMAT_R24_UNORM_X8_TYPELESS; return TRUE;
    case DXGI_FORMAT_D32_FLOAT: case DXGI_FORMAT_R32_TYPELESS: case DXGI_FORMAT_R32_FLOAT:
        *view = DXGI_FORMAT_R32_FLOAT; *hdr = TRUE; return TRUE;
    case DXGI_FORMAT_D16_UNORM: case DXGI_FORMAT_R16_TYPELESS: case DXGI_FORMAT_R16_UNORM:
        *view = DXGI_FORMAT_R16_UNORM; return TRUE;
    case DXGI_FORMAT_R8G8B8A8_TYPELESS: case DXGI_FORMAT_R8G8B8A8_UNORM: case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        *view = DXGI_FORMAT_R8G8B8A8_UNORM; return TRUE;
    case DXGI_FORMAT_R8G8B8A8_SNORM:
        *view = f; return TRUE;
    case DXGI_FORMAT_B8G8R8A8_TYPELESS: case DXGI_FORMAT_B8G8R8A8_UNORM: case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        *view = DXGI_FORMAT_B8G8R8A8_UNORM; return TRUE;
    case DXGI_FORMAT_R10G10B10A2_TYPELESS: case DXGI_FORMAT_R10G10B10A2_UNORM:
        *view = DXGI_FORMAT_R10G10B10A2_UNORM; return TRUE;
    case DXGI_FORMAT_R16G16B16A16_TYPELESS: case DXGI_FORMAT_R16G16B16A16_FLOAT:
        *view = DXGI_FORMAT_R16G16B16A16_FLOAT; *hdr = TRUE; return TRUE;
    case DXGI_FORMAT_R32G32B32A32_TYPELESS: case DXGI_FORMAT_R32G32B32A32_FLOAT:
        *view = DXGI_FORMAT_R32G32B32A32_FLOAT; *hdr = TRUE; return TRUE;
    case DXGI_FORMAT_R11G11B10_FLOAT: case DXGI_FORMAT_R9G9B9E5_SHAREDEXP:
        *view = f; *hdr = TRUE; return TRUE;
    case DXGI_FORMAT_R16G16_TYPELESS: case DXGI_FORMAT_R16G16_FLOAT:
        *view = DXGI_FORMAT_R16G16_FLOAT; *hdr = TRUE; return TRUE;
    case DXGI_FORMAT_R16G16_UNORM: case DXGI_FORMAT_R16G16_SNORM: case DXGI_FORMAT_R8G8_UNORM:
    case DXGI_FORMAT_R8G8_SNORM: case DXGI_FORMAT_R8_UNORM: case DXGI_FORMAT_A8_UNORM:
        *view = f; return TRUE;
    case DXGI_FORMAT_R8G8_TYPELESS: *view = DXGI_FORMAT_R8G8_UNORM; return TRUE;
    case DXGI_FORMAT_R8_TYPELESS: *view = DXGI_FORMAT_R8_UNORM; return TRUE;
    case DXGI_FORMAT_R16_FLOAT: case DXGI_FORMAT_R32G32_FLOAT:
        *view = f; *hdr = TRUE; return TRUE;
    case DXGI_FORMAT_R32G32_TYPELESS: *view = DXGI_FORMAT_R32G32_FLOAT; *hdr = TRUE; return TRUE;
    default:
        return FALSE;
    }
}

BOOL debug_view_record(ID3D12GraphicsCommandList *list, const struct debug_view_pick *pick, ID3D12Resource *dst,
                       UINT dst_w, UINT dst_h)
{
    static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
    ID3D12Device *device = rebox_shared_device();
    D3D12_SHADER_RESOURCE_VIEW_DESC srv = { .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING };
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav = { .Format = DXGI_FORMAT_R8G8B8A8_UNORM,
                                             .ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D };
    D3D12_RESOURCE_BARRIER barrier = { .Type = D3D12_RESOURCE_BARRIER_TYPE_UAV };
    D3D12_CPU_DESCRIPTOR_HANDLE cpu;
    D3D12_GPU_DESCRIPTOR_HANDLE gpu;
    const D3D12_RESOURCE_DESC *d = &pick->desc;
    UINT slot;
    BOOL hdr;
    struct
    {
        UINT dst_w, dst_h, src_w, src_h, mode;
        float scale, power;
        UINT channel;
    } constants;

    InitOnceExecuteOnce(&once, init_pipeline, NULL, NULL);
    if (!pipeline || d->Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || d->SampleDesc.Count != 1
            || !read_format(d->Format, &srv.Format, &hdr))
    {
        static LONG logged;
        if (InterlockedIncrement(&logged) <= 4)
            rebox_log("d3d12_xs: REBOX_DEBUG_VIEW: a %llux%u target of dimension %u, format %u, %u samples can't be "
                      "shown", (unsigned long long)d->Width, d->Height, d->Dimension, d->Format, d->SampleDesc.Count);
        return FALSE;
    }
    if (d->DepthOrArraySize > 1)
    {
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
        srv.Texture2DArray = (D3D12_TEX2D_ARRAY_SRV){ 0, 1, 0, 1, 0, 0.0f };
    }
    else
    {
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Texture2D = (D3D12_TEX2D_SRV){ 0, 1, 0, 0.0f };
    }
    slot = (UINT)InterlockedIncrement(&next_slot) % RING;
    cpu = ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(heap);
    gpu = ID3D12DescriptorHeap_GetGPUDescriptorHandleForHeapStart(heap);
    cpu.ptr += (SIZE_T)slot * 2 * increment;
    gpu.ptr += (UINT64)slot * 2 * increment;
    ID3D12Device_CreateShaderResourceView(device, pick->res, &srv, cpu);
    cpu.ptr += increment;
    ID3D12Device_CreateUnorderedAccessView(device, dst, NULL, &uav, cpu);

    constants.dst_w = dst_w;
    constants.dst_h = dst_h;
    constants.src_w = (UINT)d->Width;
    constants.src_h = d->Height;
    constants.mode = pick->depth ? (depth_forward ? 3 : 2) : hdr ? 1 : 0;
    constants.scale = colour_scale;
    constants.power = decades;
    constants.channel = channel;

    ID3D12GraphicsCommandList_SetDescriptorHeaps(list, 1, &heap);
    ID3D12GraphicsCommandList_SetComputeRootSignature(list, root_signature);
    ID3D12GraphicsCommandList_SetPipelineState(list, pipeline);
    ID3D12GraphicsCommandList_SetComputeRootDescriptorTable(list, 0, gpu);
    ID3D12GraphicsCommandList_SetComputeRoot32BitConstants(list, 1, 8, &constants, 0);
    ID3D12GraphicsCommandList_Dispatch(list, (dst_w + 7) / 8, (dst_h + 7) / 8, 1);
    barrier.UAV.pResource = dst;
    ID3D12GraphicsCommandList_ResourceBarrier(list, 1, &barrier);
    return TRUE;
}
