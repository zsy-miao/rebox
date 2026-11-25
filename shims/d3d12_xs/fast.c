/* Program names, and the detail REBOX_FAST gives up for speed.
 *
 * Every console shader container carries its entry point's name in the PSV0 part (for example
 * "MS_VS_Transform_eb34a03f_Wrapped": a stage prefix, a name, a hash and "_Wrapped"). The title's
 * shader archive has about 4,300 of them in clear text, compute included. REBOX_FAST picks programs
 * to leave out by name: a left-out program gets the fallback shaders a missing translation gets
 * (d3d12_xs.c), so it is never compiled by AnyPS5 and its draws and dispatches do nothing, at no
 * per-draw binding cost.
 *
 * REBOX_FAST=1 (or "on") leaves out the categories below marked default. REBOX_FAST=max adds the
 * risky ones. REBOX_FAST=<category>,... picks categories by name. Matching is by substring of the
 * entry name, case-sensitive.
 *   REBOX_SKIP_PROGRAMS=<substring>,...  more programs to leave out, with or without REBOX_FAST
 *   REBOX_KEEP_PROGRAMS=<substring>,...  programs never left out, whatever matches them
 *   REBOX_PROGRAM_NAMES=1  log every program's name once, with the id the census and trace give it
 *
 * "Risky" means the program writes something a later pass multiplies the scene by or reads as
 * lighting (a shadow mask, a transmittance volume, indirect light): left out, that input keeps
 * stale or zero contents, and the lit picture can go dark. Default categories only take away
 * something added on top of the scene, or a picture of its own (a reflection, particles, hair). The
 * geometry passes (depth prepass, G-buffer, forward, the composites) match nothing here. */

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/rebox_log.h"
#include "fast.h"

struct rule
{
    const char *pattern;  /* substring of the entry name */
    const char *category;
    const char *reason;
};

/* Categories: whether REBOX_FAST=1 takes them. */
static const struct { const char *name; BOOL by_default; } categories[] =
{
    { "rays", TRUE }, { "hair", TRUE }, { "volumetrics", TRUE }, { "reflections", TRUE }, { "particles", TRUE },
    { "fog", FALSE }, { "gi", FALSE }, { "ao", FALSE }, { "sss", FALSE }, { "shadows", FALSE },
};

static const struct rule rules[] =
{
    /* rays: ray tracing isn't translated, and acceleration structure builds are skipped
     * (proxy.c), so these trace an acceleration structure that was never built. DispatchRays and
     * ray ExecuteIndirects are already skipped (proxy.c). */
    { "TraceRays_BVH", "rays", "screen-probe GI rays through the BVH, which is never built" },
    { "RTShadowTrace", "rays", "ray-traced sun shadows through an unbuilt BVH (the mask keeps old contents)" },
    { "RTDistantShadowTrace", "rays", "ray-traced distant shadows through an unbuilt BVH" },
    { "RTDistantLODLightShadow", "rays", "ray-traced LOD light shadows through an unbuilt BVH" },
    { "TraceShadowRays", "rays", "shadow rays through an unbuilt BVH" },
    { "RaytraceShadows", "rays", "shadow rays through an unbuilt BVH" },
    { "RaytraceVBuffer", "rays", "ray-traced visibility buffer through an unbuilt BVH" },
    { "IrradianceCache_ShadowTrace", "rays", "irradiance cache shadow rays through an unbuilt BVH" },
    { "ShadowRaytracing", "rays", "laser light shadow rays through an unbuilt BVH" },
    { "CollisionRaytraced", "rays", "ray-traced particle collision through an unbuilt BVH" },
    { "PPR_Raytrace", "rays", "planar reflection rays (reflections only)" },

    /* hair: strand hair. The forks leave the same programs out by default (they don't translate,
     * or drew lines across the screen); hair falls back to the head's mesh. */
    { "HairStrand", "hair", "strand hair geometry, G-buffer and V-buffer passes" },
    { "InterpolateStrand", "hair", "strand hair interpolation (compute)" },
    { "VisHair", "hair", "strand hair visibility (compute)" },
    { "GenerateHair", "hair", "strand hair index and vertex buffers (compute)" },
    { "PS_Smh", "hair", "strand hair shadows and entity mask" },

    /* volumetrics: light shafts and volumes drawn additively over the scene, and clouds. */
    { "AtmosphericVolumetric", "volumetrics", "volumetric light shafts per light, additive" },
    { "UnderwaterVolumetric", "volumetrics", "underwater volumetric light shafts per light, additive" },
    { "FogVolumeProxy", "volumetrics", "fog volume proxies drawn over the scene" },
    { "FireVolume", "volumetrics", "volumetric fire" },
    { "CalculateFireVolumes", "volumetrics", "volumetric fire (compute; also on a fork's skip list)" },
    { "CloudMultiScatteringCache", "volumetrics", "cloud multiple scattering (sky only)" },
    { "CloudDensityField", "volumetrics", "cloud density (sky only)" },
    { "CloudMap", "volumetrics", "cloud map (sky only)" },

    /* reflections: separate pictures of the scene for reflective surfaces only. */
    { "MirrorReflection", "reflections", "the mirror pass: the scene drawn again, mirrored" },
    { "MIRROR_PBR_MATERIAL", "reflections", "lighting of the mirror pass" },
    { "ResolveToMirrorGBuffer", "reflections", "the mirror pass's G-buffer resolve" },
    { "ReflectionGBuffer", "reflections", "reflection probe G-buffer (the scene drawn again for a probe)" },
    { "_SSR", "reflections", "screen-space reflection rays" },
    { "GBufferSSR", "reflections", "screen-space reflection resolve" },

    /* particles: drawn over the scene with blending. */
    { "ptfx_", "particles", "particle effects" },
    { "corona", "particles", "light coronas (additive sprites)" },
    { "Weather_Particle", "particles", "weather particles" },

    /* Risky (REBOX_FAST=max or by name). */
    { "TransmissionVolumetrics", "fog", "volumetric fog injection (the scene is multiplied by its transmittance)" },
    { "GatherRaymarchedLocalLights", "fog", "raymarched local light in-scattering" },
    { "ReconstructRaymarch", "fog", "volumetric fog reconstruction" },
    { "ResolveVolumetrics", "fog", "volumetric fog resolve" },
    { "AccumulateLocalLightInscattering", "fog", "local light in-scattering volume" },
    { "RayMiss_Volumetrics", "fog", "volumetric sky in-scattering" },
    { "ScreenProbes_", "gi", "screen-probe GI (indirect light keeps old contents; a fork puts a constant there)" },
    { "ScreenProbe_", "gi", "screen-probe GI" },
    { "IrradianceCache_", "gi", "irradiance cache GI" },
    { "CS_Primary_", "gi", "primary GI probes" },
    { "FinalGather_", "gi", "GI final gather" },
    { "ScreenSpaceBentCones", "ao", "screen-space ambient occlusion (bent cones)" },
    { "SSSResolve", "sss", "subsurface scattering resolve" },
    { "SSSPrefilter", "sss", "subsurface scattering prefilter" },
    { "SSSDilate", "sss", "subsurface scattering dilation" },
    { "CascadeShadows", "shadows", "cascaded sun shadow maps (the map keeps old contents)" },
    { "ShadowRender", "shadows", "shadow map rendering" },
    { "LocalShadow", "shadows", "local light shadow maps" },
};

static BOOL category_on[ARRAYSIZE(categories)];
static char *extra_skip, *keep;
static BOOL names_logged;

static BOOL CALLBACK fast_init(INIT_ONCE *once, void *param, void **ctx)
{
    char *fast = rebox_getenv("REBOX_FAST"), *names = rebox_getenv("REBOX_PROGRAM_NAMES");
    char on[160] = "";
    int at = 0;

    names_logged = names && *names && *names != '0';
    free(names);
    extra_skip = rebox_getenv("REBOX_SKIP_PROGRAMS");
    keep = rebox_getenv("REBOX_KEEP_PROGRAMS");
    if (fast && *fast && strcmp(fast, "0"))
    {
        BOOL all = !strcmp(fast, "max"), defaults = all || !strcmp(fast, "1") || !strcmp(fast, "on");
        for (UINT i = 0; i < ARRAYSIZE(categories); i++)
        {
            const char *hit = strstr(fast, categories[i].name);
            SIZE_T len = strlen(categories[i].name);
            category_on[i] = all || (defaults && categories[i].by_default)
                    || (hit && (hit == fast || hit[-1] == ',') && (!hit[len] || hit[len] == ','));
            if (category_on[i] && at < (int)sizeof(on) - 16)
                at += snprintf(on + at, sizeof(on) - at, "%s%s", at ? "," : "", categories[i].name);
        }
    }
    if (*on || extra_skip)
        rebox_log("d3d12_xs: REBOX_FAST: leaving out programs in %s%s%s%s%s", *on ? on : "no category",
                  extra_skip ? ", and those matching " : "", extra_skip ? extra_skip : "",
                  keep ? "; keeping those matching " : "", keep ? keep : "");
    free(fast);
    return TRUE;
}

/* The NUL-terminated name at offset in PSV0's string table, if it is one. */
static BOOL psv0_name(const BYTE *body, UINT32 size, char *name, SIZE_T name_size)
{
    UINT32 info_size, at, strings, offset;

    /* PSVRuntimeInfo3 (52 bytes) names the entry function by an offset into the string table,
     * which follows the resource bindings. */
    if (size < 4 || (info_size = *(const UINT32 *)body) < 52 || 4ull + info_size + 4 > size)
        return FALSE;
    offset = *(const UINT32 *)(body + 4 + 48);
    at = 4 + info_size;
    {
        UINT32 count = *(const UINT32 *)(body + at);
        at += 4;
        if (count)
        {
            UINT32 stride;
            if (at + 4 > size)
                return FALSE;
            stride = *(const UINT32 *)(body + at);
            at += 4;
            if ((UINT64)count * stride > size - at)
                return FALSE;
            at += count * stride;
        }
    }
    if (at + 4 > size)
        return FALSE;
    strings = *(const UINT32 *)(body + at);
    at += 4;
    if (strings > size - at || offset >= strings)
        return FALSE;
    {
        const char *s = (const char *)body + at + offset;
        SIZE_T len = strnlen(s, strings - offset);
        if (!len || len == strings - offset || len >= name_size)
            return FALSE;
        memcpy(name, s, len + 1);
    }
    return TRUE;
}

BOOL fast_program_name(const void *container, SIZE_T size, char *name, SIZE_T name_size)
{
    const BYTE *d = container;
    UINT32 parts;

    if (!d || size < 32 || memcmp(d, "DXBC", 4))
        return FALSE;
    parts = *(const UINT32 *)(d + 28);
    for (UINT32 i = 0; i < parts && 32 + 4ull * i + 4 <= size; i++)
    {
        UINT32 off = *(const UINT32 *)(d + 32 + 4 * i), chunk;
        if (off > size - 8 || memcmp(d + off, "PSV0", 4))
            continue;
        chunk = *(const UINT32 *)(d + off + 4);
        if (chunk > size - off - 8)
            return FALSE;
        return psv0_name(d + off + 8, chunk, name, name_size);
    }
    return FALSE;
}

/* Whether a comma-separated list of substrings has one in name. */
static BOOL list_matches(const char *list, const char *name)
{
    char item[128];
    while (list && *list)
    {
        const char *end = strchr(list, ',');
        SIZE_T len = end ? (SIZE_T)(end - list) : strlen(list);
        if (len && len < sizeof(item))
        {
            memcpy(item, list, len);
            item[len] = 0;
            if (strstr(name, item))
                return TRUE;
        }
        list = end ? end + 1 : NULL;
    }
    return FALSE;
}

/* Programs already logged as left out, so each is logged once. */
static UINT64 logged_hashes[4096];
static SRWLOCK logged_lock = SRWLOCK_INIT;

static BOOL first_time(const char *name)
{
    UINT64 h = 0xcbf29ce484222325ull;
    BOOL first = TRUE;
    for (const char *c = name; *c; c++)
        h = (h ^ (BYTE)*c) * 0x100000001b3ull;
    h |= 1;
    AcquireSRWLockExclusive(&logged_lock);
    for (UINT n = 0, i = (UINT)(h % ARRAYSIZE(logged_hashes)); n < ARRAYSIZE(logged_hashes);
         n++, i = (i + 1) % ARRAYSIZE(logged_hashes))
    {
        if (logged_hashes[i] == h)
        {
            first = FALSE;
            break;
        }
        if (!logged_hashes[i])
        {
            logged_hashes[i] = h;
            break;
        }
    }
    ReleaseSRWLockExclusive(&logged_lock);
    return first;
}

const char *fast_skip_reason(const void *container, SIZE_T size, const char *kind)
{
    static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
    static volatile LONG left_out;
    const char *reason = NULL, *category = NULL;
    char name[256];

    InitOnceExecuteOnce(&once, fast_init, NULL, NULL);
    if (!fast_program_name(container, size, name, sizeof(name)))
        return NULL;
    if (keep && list_matches(keep, name))
        return NULL;
    for (UINT i = 0; i < ARRAYSIZE(rules) && !reason; i++)
    {
        if (!strstr(name, rules[i].pattern))
            continue;
        for (UINT c = 0; c < ARRAYSIZE(categories); c++)
            if (category_on[c] && !strcmp(categories[c].name, rules[i].category))
            {
                reason = rules[i].reason;
                category = rules[i].category;
            }
    }
    if (!reason && extra_skip && list_matches(extra_skip, name))
        reason = "REBOX_SKIP_PROGRAMS", category = "named";
    /* Compute programs named as writers of indirect arguments (culling, compaction, argument
     * set-up) are kept: left out, every mesh ExecuteIndirect they feed reads stale records
     * (producers.c, which makes them essential at runtime; a program left out here never compiles,
     * so it can't be found there). */
    if (reason && !strcmp(kind, "cs"))
    {
        static const char *const producer_names[] = { "BatchCull", "CullMeshlets", "CulledMeshlet", "VisibleMeshlets",
                "VisibleCount", "PackInstances", "IndirectArgs", "DispatchArgs", "InitArgs", "IndirectDrawInfo",
                "DrawBatch" };
        for (UINT i = 0; i < ARRAYSIZE(producer_names); i++)
            if (strstr(name, producer_names[i]))
            {
                if (first_time(name))
                    rebox_log("d3d12_xs: REBOX_FAST: cs program %s kept although %s (%s) would leave it out: it writes "
                              "indirect arguments (named %s)", name, category, reason, producer_names[i]);
                return NULL;
            }
    }
    if (reason && first_time(name))
    {
        LONG n = InterlockedIncrement(&left_out);
        rebox_log("d3d12_xs: REBOX_FAST: %s program %s left out (%s: %s); %ld so far", kind, name, category, reason, n);
    }
    return reason;
}

/* Id -> name, for the census and the trace: open addressing, entries never removed. */
static struct { volatile UINT64 id; char *name; } id_names[16384];

void fast_name_id(UINT64 id, const void *container, SIZE_T size)
{
    static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
    char name[256];

    InitOnceExecuteOnce(&once, fast_init, NULL, NULL);
    if (!id || !fast_program_name(container, size, name, sizeof(name)))
        return;
    for (UINT n = 0, i = (UINT)(id % ARRAYSIZE(id_names)); n < ARRAYSIZE(id_names); n++, i = (i + 1) % ARRAYSIZE(id_names))
    {
        UINT64 at = id_names[i].id;
        if (at == id)
            return;
        if (!at && !InterlockedCompareExchange64((volatile LONG64 *)&id_names[i].id, (LONG64)id, 0))
        {
            id_names[i].name = _strdup(name);
            if (names_logged)
                rebox_log("d3d12_xs: program %016llx is %s", (unsigned long long)id, name);
            return;
        }
    }
}

const char *fast_id_name(UINT64 id)
{
    if (!id)
        return NULL;
    for (UINT n = 0, i = (UINT)(id % ARRAYSIZE(id_names)); n < ARRAYSIZE(id_names); n++, i = (i + 1) % ARRAYSIZE(id_names))
    {
        UINT64 at = id_names[i].id;
        if (!at)
            return NULL;
        if (at == id)
            return id_names[i].name; /* NULL while its writer is still copying it */
    }
    return NULL;
}

/* For tests and tools: a container's program name (as fast_program_name), and whether REBOX_FAST
 * leaves it out (its reason, or NULL). */
BOOL WINAPI ReboxProgramName(const void *container, SIZE_T size, char *name, SIZE_T name_size)
{
    return fast_program_name(container, size, name, name_size);
}

const char *WINAPI ReboxProgramLeftOut(const void *container, SIZE_T size)
{
    return fast_skip_reason(container, size, "test");
}
