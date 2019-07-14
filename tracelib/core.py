from .vector import Vec

from dataclasses import dataclass
from PIL import Image
import math, multiprocessing, time

NO_INTERSECTION = (None, None)

@dataclass
class Camera:
    fov: float
    res: list # w, h
    pos: Vec  # x, y, z
    rot: Vec  # roll, pan, tilt

@dataclass
class Scene:
    materials: dict
    objs: list
    lights: list
    background_color: Vec

@dataclass
class Options:
    max_depth: int
    proc_count: int = 4
    bias: float = 0.00001

@dataclass
class Material:
    reflective: bool
    refractive: bool
    spec_exponent: float
    diffuse_color: Vec
    Kd: float   # Diffuse reflection
    Ks: float   # Specular reflection
    ior: float

@dataclass
class Light:
    pos: Vec
    intensity: float

min_pos = lambda *args: min([x for x in args if x >= 0])
clamp = lambda x, m, M: max(min(x, M), m)
sgn = lambda n: math.copysign(1, n)

def rot(v, cos_rx, cos_ry, cos_rz, sin_rx, sin_ry, sin_rz):
    # `rot` uses cached trig values for a nice bump in efficiency

    return Vec(
        cos_rz * (cos_ry * v[0] - sin_ry * (-sin_rx * v[1] + cos_rx * v[2])) + sin_rz * (cos_rx * v[1] + sin_rx * v[2]),
        -sin_rz * (cos_ry * v[0] - sin_ry * (-sin_rx * v[1] + cos_rx * v[2])) + cos_rz * (cos_rx * v[1] + sin_rx * v[2]),
        sin_ry * v[0] + cos_ry * (-sin_rx * v[1] + cos_rx * v[2])
    )

def check_intersection(pos, v, objs):
    min_d = None
    min_p = None
    min_o = None

    for obj in objs:
        (d, p) = obj.intersection(pos, v)
        if (d is not None and (min_d == None or d < min_d)):
            min_d = d
            min_p = p
            min_o = obj

    return (min_d, min_p, min_o)

def reflect(d, n):
    return d - n * 2 * d.dot(n)

def refract(d, n, ior):
    cosi = clamp(d.dot(n), -1, 1)
    etai = 1
    etat = ior

    if (cosi < 0):
        cosi = -cosi
    else:
        etai, etat = etat, etai
        n = -n

    eta = etai / etat
    k = 1 - eta ** 2 * (1 - cosi ** 2)

    if (k < 0):
        return Vec(0, 0, 0)
    else:
        return d * eta + n * (eta * cosi - math.sqrt(k))

def fresnel(d, n, ior):
    cosi = clamp(d.dot(n), -1, 1)
    etai = 1
    etat = ior

    if (cosi > 0):
        etai, etat = etat, etai

    sint = etai / etat * math.sqrt(max(1 - cosi * cosi, 0))

    if (sint >= 1):
        return 1.0
    else:
        cost = math.sqrt(max(1 - sint * sint, 0))
        cosi = abs(cosi)
        Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost))
        Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost))
        return (Rs * Rs + Rp * Rp) / 2.0

def cast_ray(pos, d, scene, options, depth=0):
    hit_clr = scene.background_color

    if (depth > options.max_depth):
        return hit_clr

    (min_d, min_p, min_o) = check_intersection(pos, d, scene.objs)

    if (min_d != None):
        material = scene.materials[min_o.material]

        n = min_o.normal(pos, min_p)

        # TODO reflection is broken, refraction is not
        if (material.reflective):
            kr = fresnel(d, n, material.ior)

            if (material.refractive):
                reflection_dir = ~reflect(d, n)
                reflection_orig = min_p - n * options.bias if (reflection_dir.dot(n) < 0) else min_p + n * options.bias
                reflection_clr = cast_ray(reflection_orig, reflection_dir, scene, options, depth=depth+1)
                refraction_dir = ~refract(d, n, material.ior)
                refraction_orig = min_p - n * options.bias if (refraction_dir.dot(n) < 0) else min_p + n * options.bias
                refraction_clr = cast_ray(refraction_orig, refraction_dir, scene, options, depth=depth+1)
                hit_clr = reflection_clr * kr + refraction_clr * (1 - kr)
            else:
                reflection_dir = reflect(d, n)
                reflection_orig = min_p + n * options.bias if (reflection_dir.dot(n) < 0) else min_p - n * options.bias
                reflection_clr = cast_ray(reflection_orig, reflection_dir, scene, options, depth=depth+1)
                hit_clr = reflection_clr * kr

        else:
            light_amt = 0
            spec_clr = Vec(0.0, 0.0, 0.0)
            shadow_orig = min_p + n * options.bias if (d.dot(n) < 0) else min_p - n * options.bias

            for light in scene.lights:
                vec = light.pos - shadow_orig
                ld2 = vec.mag2()
                vec = ~vec
                ldn = max(vec.dot(n), 0)
                (s_d, s_p, s_o) = check_intersection(shadow_orig, vec, scene.objs)
                if (s_d == None or (s_d ** 2) >= ld2):
                    light_amt += light.intensity * ldn
                reflection_dir = reflect(-vec, n)
                spec_clr += pow(max(-reflection_dir.dot(d), 0), material.spec_exponent) * light.intensity

            hit_clr = material.diffuse_color * light_amt * material.Kd + spec_clr * material.Ks

    return hit_clr

def render_worker(scene, camera, options, in_queue, out_queue):
    while True:
        y = in_queue.get()

        line = [(0, 0, 0) for j in range(camera.res[0])]

        #if (y % 2):
        #    out_queue.put((y, line))
        #    continue
        
        for x in range(camera.res[0]):
            vec = ~rot(Vec(
                (2 * (x + 0.5) / camera.res[0] - 1) * camera.aspect_ratio * camera.scale,
                (1 - 2 * (y + 0.5) / camera.res[1]) * camera.scale,
                1.0
            ), camera.cos_rx, camera.cos_ry, camera.cos_rz, camera.sin_rx, camera.sin_ry, camera.sin_rz)

            c = cast_ray(camera.pos, vec, scene, options)
            line[x] = tuple(int(0xff * x) for x in c)

        print(str(y).ljust(4), end=' ', flush=True)
        out_queue.put((y, line))

def render_scene(scene, camera, options):
    print('Rendering scene...')
    u = time.time()

    camera.cos_rx, camera.cos_ry, camera.cos_rz = math.cos(camera.rot[0]), math.cos(camera.rot[1]), math.cos(camera.rot[2])
    camera.sin_rx, camera.sin_ry, camera.sin_rz = math.sin(camera.rot[0]), math.sin(camera.rot[1]), math.sin(camera.rot[2])

    camera.aspect_ratio = camera.res[0] / camera.res[1]
    camera.scale = math.tan(math.radians(camera.fov / 2.0))

    line_count = camera.res[1]

    multiprocessing.freeze_support()

    job_in_queue = multiprocessing.Queue(line_count)
    job_out_queue = multiprocessing.Queue(line_count)

    processes = []

    for i in range(line_count):
        job_in_queue.put(i)

    for i in range(options.proc_count):
        proc = multiprocessing.Process(
            target = render_worker,
            args = (scene, camera, options, job_in_queue, job_out_queue)
        )

        processes.append(proc)
        proc.start()

    slices = [job_out_queue.get() for n in range(line_count)]

    for i in range(options.proc_count):
        processes[i].terminate()

    screen = [s[1] for s in sorted(slices, key=lambda i: i[0])]

    v = time.time()
    print('\nScene took {} seconds to render'.format(v - u))

    return screen

def fxaa(pixels, depthmap):
    w = len(pixels[0])
    h = len(pixels)

    dmap = [[0 for i in range(w)] for j in range(h)]

    for y in range(h):
        for x in range(w):
            max_d = 0
            p = depthmap[y][x]

            if (x > 0):
                d = abs(depthmap[y][x - 1] - p)
                if (d > max_d):
                    max_d = d
            if (x < w - 1):
                d = abs(depthmap[y][x + 1] - p)
                if (d > max_d):
                    max_d = d
            if (y > 0):
                d = abs(depthmap[y - 1][x] - p)
                if (d > max_d):
                    max_d = d
            if (y < h - 1):
                d = abs(depthmap[y + 1][x] - p)
                if (d > max_d):
                    max_d = d

            dmap[y][x] = max_d

    for y in range(h):
        for x in range(w):
            d_adj = dmap[y][x]

            if (d_adj > 0.5):
                pass # it's an edge maybe?

    return dmap

def image_from_pixels(pixels):
    dim = (len(pixels[0]), len(pixels))
    img = Image.new('RGB', dim, 'black')
    pix = img.load()

    for i in range(dim[0]):
        for j in range(dim[1]):
            pix[i, j] = pixels[j][i]

    return img