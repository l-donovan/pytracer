from .vector import Vec

from dataclasses import dataclass
from PIL import Image
import math
import multiprocessing
import time

NO_INTERSECTION = (None, None)


@dataclass
class Camera:
    fov: float
    res: tuple  # w, h
    pos: Vec    # x, y, z
    rot: Vec    # roll, pan, tilt


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
    specular_exp: float
    diffuse_color: Vec
    Kd: float   # Diffuse reflection coefficient
    Ks: float   # Specular reflection coefficient
    ior: float  # Index of refraction


@dataclass
class Light:
    pos: Vec
    intensity: float


def min_positive(*args):
    return min([arg for arg in args if arg >= 0])


def clamp(num, lower, upper):
    return max(min(num, upper), lower)


def sgn(num):
    return (num > 0) - (num < 0)


def rot(v, cos_rx, cos_ry, cos_rz, sin_rx, sin_ry, sin_rz):
    # `rot` uses cached trig values for a nice bump in efficiency

    return Vec(
        cos_rz * (cos_ry * v[0] - sin_ry * (-sin_rx * v[1] + cos_rx * v[2])) + sin_rz * (cos_rx * v[1] + sin_rx * v[2]),
        -sin_rz * (cos_ry * v[0] - sin_ry * (-sin_rx * v[1] + cos_rx * v[2])) + cos_rz * (
                cos_rx * v[1] + sin_rx * v[2]),
        sin_ry * v[0] + cos_ry * (-sin_rx * v[1] + cos_rx * v[2])
    )


def check_intersection(origin, v, objs):
    hit_dist, hit_pos, hit_obj = None, None, None

    for obj in objs:
        dist, pos = obj.intersection(origin, v)
        if dist is not None and (hit_dist is None or dist < hit_dist):
            hit_dist, hit_pos, hit_obj = dist, pos, obj

    return hit_dist, hit_pos, hit_obj


def reflect(d, n):
    return d - n * 2 * d.dot(n)


def refract(d, n, ior):
    cosi = clamp(d.dot(n), -1, 1)
    etai = 1
    etat = ior

    if cosi < 0:
        cosi = -cosi
    else:
        etai, etat = etat, etai
        n = -n

    eta = etai / etat
    k = 1 - eta ** 2 * (1 - cosi ** 2)

    if k < 0:
        return Vec(0, 0, 0)
    else:
        return d * eta + n * (eta * cosi - math.sqrt(k))


def fresnel(d, n, ior):
    cosi = clamp(d.dot(n), -1, 1)
    etai = 1
    etat = ior

    if cosi > 0:
        etai, etat = etat, etai

    sint = etai / etat * math.sqrt(max(1 - cosi * cosi, 0))

    if sint >= 1:
        return 1.0
    else:
        cost = math.sqrt(max(1 - sint * sint, 0))
        cosi = abs(cosi)
        rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost))
        rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost))
        return (rs * rs + rp * rp) / 2.0


def cast_ray(pos, d, scene, options, depth=0, dist=None):
    hit_color = scene.background_color

    if depth > options.max_depth:
        return hit_color, dist

    hit_dist, hit_pos, hit_obj = check_intersection(pos, d, scene.objs)

    if hit_dist is None:
        total_dist = dist
    elif dist is None:
        total_dist = hit_dist
    else:
        total_dist = dist + hit_dist

    if hit_dist is not None:
        material = scene.materials[hit_obj.material]

        n = hit_obj.normal(pos, hit_pos)

        # TODO reflection is broken, refraction is not
        if material.reflective:
            kr = fresnel(d, n, material.ior)

            if material.refractive:
                reflection_dir = reflect(d, n).norm()
                reflection_origin = hit_pos + n * options.bias * sgn(reflection_dir.dot(n))
                reflection_color, total_dist = \
                    cast_ray(reflection_origin, reflection_dir, scene, options, depth=depth + 1, dist=total_dist)
                refraction_dir = refract(d, n, material.ior).norm()
                refraction_origin = hit_pos + n * options.bias * sgn(refraction_dir.dot(n))
                refraction_color, total_dist = \
                    cast_ray(refraction_origin, refraction_dir, scene, options, depth=depth + 1, dist=total_dist)
                hit_color = reflection_color * kr + refraction_color * (1 - kr)
            else:
                reflection_dir = reflect(d, n).norm()
                reflection_origin = hit_pos - n * options.bias * sgn(reflection_dir.dot(n))
                reflection_color, total_dist = \
                    cast_ray(reflection_origin, reflection_dir, scene, options, depth=depth + 1, dist=total_dist)
                hit_color = reflection_color * kr
        else:
            light_amt = 0
            specular_color = Vec(0.0, 0.0, 0.0)
            shadow_origin = hit_pos + n * options.bias if (d.dot(n) < 0) else hit_pos - n * options.bias

            for light in scene.lights:
                vec = light.pos - shadow_origin
                ld2 = vec.mag2()
                vec = vec.norm()
                ldn = max(vec.dot(n), 0)
                shadow_dist, shadow_pos, shadow_obj = check_intersection(shadow_origin, vec, scene.objs)
                if shadow_dist is None or (shadow_dist ** 2) >= ld2:
                    light_amt += light.intensity * ldn
                reflection_dir = reflect(-vec, n)
                specular_color += pow(max(-reflection_dir.dot(d), 0), material.specular_exp) * light.intensity

            hit_color = material.diffuse_color * light_amt * material.Kd + specular_color * material.Ks

    return hit_color, total_dist



def render_worker(scene, camera, options, in_queue, out_queue):
    while True:
        y = in_queue.get()

        line_color = [(0, 0, 0) for j in range(camera.res[0])]
        line_dist = [None for j in range(camera.res[0])]

        for x in range(camera.res[0]):
            vec = rot(Vec(
                (2 * (x + 0.5) / camera.res[0] - 1) * camera.aspect_ratio * camera.scale,
                (1 - 2 * (y + 0.5) / camera.res[1]) * camera.scale,
                1.0
            ), camera.cos_rx, camera.cos_ry, camera.cos_rz, camera.sin_rx, camera.sin_ry, camera.sin_rz).norm()

            color, dist = cast_ray(camera.pos, vec, scene, options)

            line_color[x] = (int(0xff * color[0]), int(0xff * color[1]), int(0xff * color[2]))
            line_dist[x] = dist if dist is not None else 100000

        print(str(y).ljust(4), end=' ', flush=True)
        out_queue.put((y, line_color, line_dist))



def render_scene(scene, camera, options):
    print('Rendering scene...')
    u = time.time()

    camera.cos_rx, camera.cos_ry, camera.cos_rz, camera.sin_rx, camera.sin_ry, camera.sin_rz = \
        math.cos(camera.rot[0]), math.cos(camera.rot[1]), math.cos(camera.rot[2]), \
        math.sin(camera.rot[0]), math.sin(camera.rot[1]), math.sin(camera.rot[2])

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
            target=render_worker,
            args=(scene, camera, options, job_in_queue, job_out_queue)
        )

        processes.append(proc)
        proc.start()

    slices = sorted([job_out_queue.get() for n in range(line_count)], key=lambda k: k[0])

    for i in range(options.proc_count):
        processes[i].terminate()

    screen = [s[1] for s in slices]
    depth = [s[2] for s in slices]

    v = time.time()
    print('\nScene took {} seconds to render'.format(v - u))

    q = fxaa(screen, depth)
    # print(q)

    return screen


def fxaa(pixels, depthmap):
    # Anti-aliasing tk

    w = len(pixels[0])
    h = len(pixels)

    dmap = [[0 for i in range(w)] for j in range(h)]

    for y in range(h):
        for x in range(w):
            max_d = 0
            p = depthmap[y][x]

            if x > 0:
                d = abs(depthmap[y][x - 1] - p)
                if d > max_d:
                    max_d = d
            if x < w - 1:
                d = abs(depthmap[y][x + 1] - p)
                if d > max_d:
                    max_d = d
            if y > 0:
                d = abs(depthmap[y - 1][x] - p)
                if d > max_d:
                    max_d = d
            if y < h - 1:
                d = abs(depthmap[y + 1][x] - p)
                if d > max_d:
                    max_d = d

            dmap[y][x] = max_d

    for y in range(h):
        for x in range(w):
            diff = dmap[y][x]

            if diff > 10:
                pass  # it's an edge maybe?

    return dmap


def image_from_pixels(pixels):
    dim = (len(pixels[0]), len(pixels))
    img = Image.new('RGB', dim, 'black')
    pix = img.load()

    for i in range(dim[0]):
        for j in range(dim[1]):
            pix[i, j] = pixels[j][i]

    return img


STD_MATERIALS = {
    'glass': Material(
        reflective=True,
        refractive=True,
        spec_exponent=25,
        diffuse_color=Vec(0.2, 0.2, 0.2),
        Kd=0.8,
        Ks=0.2,
        ior=1.52
    ),
    'glossy': Material(
        reflective=False,
        refractive=False,
        spec_exponent=25,
        diffuse_color=Vec(0.8, 0.7, 0.2),
        Kd=0.8,
        Ks=0.2,
        ior=1.0
    ),
    'rubber': Material(
        reflective=False,
        refractive=False,
        spec_exponent=25,
        diffuse_color=Vec(0.2, 0.2, 0.2),
        Kd=0.8,
        Ks=0.2,
        ior=1.3
    ),
    'floor': Material(
        reflective=False,
        refractive=False,
        spec_exponent=1,
        diffuse_color=Vec(0.8, 0.8, 0.8),
        Kd=0.8,
        Ks=0.2,
        ior=1.3
    ),
    'mirror': Material(
        reflective=True,
        refractive=False,
        spec_exponent=25,
        diffuse_color=Vec(0.1, 0.1, 0.1),
        Kd=1.0,
        Ks=0.2,
        ior=0.01
    )
}

# Uses standard colors in opencv::viz
STD_COLORS = {
    'black':    Vec(0.0, 0.0, 0.0),
    'white':    Vec(1.0, 1.0, 1.0),
    'red':      Vec(1.0, 0.0, 0.0),
    'green':    Vec(0.0, 1.0, 0.0),
    'blue':     Vec(0.0, 0.0, 1.0),
    'yellow':   Vec(1.0, 1.0, 0.0),
    'cyan':     Vec(0.0, 1.0, 1.0),
    'magenta':  Vec(1.0, 0.0, 1.0),
    'grey':     Vec(0.5, 0.5, 0.5),
    'navy':     Vec(0.0, 0.0, 0.5),
    'olive':    Vec(0.5, 0.5, 0.0),
    'maroon':   Vec(0.5, 0.0, 0.0),
    'teal':     Vec(0.0, 0.5, 0.5),
    'purple':   Vec(0.5, 0.0, 0.5),
    'rose':     Vec(1.0, 0.0, 0.5),
    'azure':    Vec(0.0, 0.5, 1.0),
    'lime':     Vec(0.749019, 1.0, 0.0),
    'gold':     Vec(1.0, 0.843138, 0.0),
    'brown':    Vec(0.164706, 0.164706, 0.647059),
    'orange':   Vec(1.0, 0.647059, 0.0),
    'indigo':   Vec(0.294118, 0.0, 0.509804),
    'pink':     Vec(1.0, 0.752941, 0.796078),
    'cherry':   Vec(0.870588, 0.113725, 0.388235),
    'silver':   Vec(0.752941, 0.752941, 0.752941),
    'violet':   Vec(0.541176, 0.168627, 0.886275),
    'apricot':  Vec(0.984314, 0.807843, 0.694118),
    'chartreuse':   Vec(0.5, 1.0, 0.0),
    'orange-red':   Vec(1.0, 0.270588, 0.0),
    'blueberry':    Vec(0.309804, 0.52549, 0.968627),
    'raspberry':    Vec(0.890196, 0.043137, 0.360784),
    'turquoise':    Vec(0.25098, 0.878431, 0.815686),
    'amethyst':     Vec(0.6, 0.4, 0.8),
    'celestial-blue': Vec(0.286275, 0.592157, 0.815686),
}
