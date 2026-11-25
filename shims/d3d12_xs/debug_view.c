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
 * The target is copied when the pass that drew it ends (the next OMSetRenderTargets of that list,
 * or its Close), so a later pass reusing its memory doesn't hide it, and the copy is shown.
 * REBOX_DEBUG_VIEW_SNAPSHOT=0 reads the target itself at the present instead.
 *   REBOX_DEBUG_VIEW_SCALE=<f>      colour multiplier (default 1); float formats also roll off as
 *                                   x / (1 + x)
 *   REBOX_DEBUG_VIEW_CHANNEL=r|g|b|a  one channel as grey (default the colour)
 *   REBOX_DEBUG_VIEW_DEPTH=forward  depth with the far plane at 1 (default reversed: far at 0)
 *   REBOX_DEBUG_VIEW_DECADES=<n>    depth shown on a log scale over n decades (default 4)
 * Depth is white near the camera and darker with distance; texels still at the clear value are
 * dark blue. Frame dumps (REBOX_DUMP_FRAMES) show the view too. */

#define COBJMACROS
#define WIDL_C_INLINE_WRAPPERS
#include <windows.h>
#include <d3d12.h>
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
} snap;
static struct { ID3D12Resource *res; LONG at; } retired[8];
static SRWLOCK snap_lock = SRWLOCK_INIT;

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
    rebox_log("d3d12_xs: REBOX_DEBUG_VIEW=%s: a render target or depth buffer is shown in place of the presented "
              "plane (%s; depth %s, %.1f decades; colour scale %.2f)", v,
              snapshots ? "copied when its pass ends" : "read at the present", depth_forward ? "forward" : "reversed",
              decades, colour_scale);
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

static void state_barrier(ID3D12GraphicsCommandList *list, ID3D12Resource *res, UINT kind, UINT from, BOOL copy,
                          BOOL forward)
{
    if (kind == 2) /* an enhanced-barrier layout */
    {
        UINT layout = copy ? D3D12_BARRIER_LAYOUT_COPY_SOURCE : D3D12_BARRIER_LAYOUT_SHADER_RESOURCE;
        UINT sync = copy ? D3D12_BARRIER_SYNC_COPY : D3D12_BARRIER_SYNC_COMPUTE_SHADING;
        UINT access = copy ? D3D12_BARRIER_ACCESS_COPY_SOURCE : D3D12_BARRIER_ACCESS_SHADER_RESOURCE;
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
        UINT state = copy ? D3D12_RESOURCE_STATE_COPY_SOURCE : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
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

/* Records a copy of the chosen target into its snapshot (made on first use), at the end of a pass
 * that drew it. */
static void snapshot_copy(ID3D12GraphicsCommandList *list, void *target)
{
    ID3D12Resource *copy = NULL, *res = target;
    UINT kind = 1, from = D3D12_RESOURCE_STATE_COMMON;

    AcquireSRWLockExclusive(&snap_lock);
    if (snap.of != target || !snap.copy)
    {
        D3D12_HEAP_PROPERTIES heap = { D3D12_HEAP_TYPE_DEFAULT };
        D3D12_RESOURCE_DESC d = ID3D12Resource_GetDesc(res);
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
         * too, stencil plane and all); only the flags that would keep it from being read go. */
        d.Flags &= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        d.Alignment = 0;
        d.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        hr = ID3D12Device_CreateCommittedResource(rebox_shared_device(), &heap, D3D12_HEAP_FLAG_NONE, &d,
                D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource, (void **)&made);
        snap.copy = SUCCEEDED(hr) ? made : NULL;
        snap.of = target;
        snap.copied = 0;
        if (FAILED(hr))
            rebox_log("d3d12_xs: REBOX_DEBUG_VIEW: no copy of a %llux%u format %u target (%#lx); it is read at the "
                      "present instead", (unsigned long long)d.Width, d.Height, d.Format, hr);
    }
    copy = snap.copy;
    ReleaseSRWLockExclusive(&snap_lock);
    if (!copy)
        return;
    present_tracked(res, &kind, &from);
    state_barrier(list, res, kind, from, TRUE, TRUE);
    plain_barrier(list, copy, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12GraphicsCommandList_CopyResource(list, copy, res);
    plain_barrier(list, copy, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    state_barrier(list, res, kind, from, TRUE, FALSE);
    InterlockedExchange(&snap.copied, frame + 1);
}

/* ------------------------------------------------------------ recording */

void debug_view_bind(ID3D12GraphicsCommandList *list, struct debug_view_list *l, ID3D12Resource *const *rts,
                     UINT count, ID3D12Resource *depth, BOOL display)
{
    int indices[8], d = -1;
    UINT n = min(count, 8u);
    void *picked = chosen;

    /* Only if it is still the one picked: another may be anywhere in another list. */
    if (l->snapshot_res && snapshots && l->snapshot_res == picked)
        snapshot_copy(list, picked);
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
    l->snapshot_res = NULL;
    for (UINT i = 0; picked && i < n; i++)
        if (rts[i] == picked)
            l->snapshot_res = picked;
    if (picked && depth == picked)
        l->snapshot_res = picked;
}

void debug_view_draw(struct debug_view_list *l, BOOL mesh, UINT fate)
{
    fate = min(fate, DRAW_FATES - 1u);
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
    void *picked = chosen;
    if (l->snapshot_res && snapshots && l->snapshot_res == picked)
        snapshot_copy(list, picked);
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

    /* The copy made when its pass ended in this frame or the last, if there is one. */
    AcquireSRWLockExclusive(&snap_lock);
    for (UINT k = 0; k < ARRAYSIZE(retired); k++)
        if (retired[k].res && now - retired[k].at > EVICT_PRESENTS)
        {
            ID3D12Resource_Release(retired[k].res);
            retired[k].res = NULL;
        }
    if (snapshots && snap.of == pick->res && snap.copy && snap.copied >= now - 1)
    {
        ID3D12Resource_Release(pick->res);
        pick->res = snap.copy;
        ID3D12Resource_AddRef(pick->res);
        pick->snapshot = TRUE;
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
