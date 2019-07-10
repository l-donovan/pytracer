from veclib import Vec
from dataclasses import dataclass
import math, time

import multiprocessing

from PIL import Image

DEPTH_MAX = 8

@dataclass
class Camera:
    fov: list # hfov, vfov
    res: list # w, h
    pos: Vec  # x, y, z
    rot: Vec  # roll, pan, tilt

@dataclass
class Scene:
    materials: dict
    objs: list
    lights: list
    background_color: Vec
    bias: float

@dataclass
class Material:
    reflective: bool
    refractive: bool
    spec_exponent: float
    diffuse_color: Vec
    Kd: float
    Ks: float
    ior: float

@dataclass
class Light:
    pos: Vec
    intensity: float

def min_pos(*args):
    m = None
    for arg in args:
        if (((m != None and arg < m) or m == None) and arg >= 0):
            m = arg
    return m

class Sphere:
    def __init__(self, pos, radius, material):
        self.pos = pos
        self.r2 = radius ** 2
        self.material = material

    def collision(self, p, d):
        m = p - self.pos
        b = m.dot(d)
        c = m.mag2() - self.r2

        if (c > 0 and b > 0):
            return (None, None)

        discr = b ** 2 - c

        if (discr < 0):
            return (None, None)

        t = min_pos(-b - math.sqrt(discr), -b + math.sqrt(discr))
        q = p + d * t

        return (t, q)

    def normal(self, q):
        return ~(q - self.pos)

def rot(v, cos_rx, cos_ry, cos_rz, sin_rx, sin_ry, sin_rz):
    # `rot` uses cached trig values for a nice bump in efficiency
    # return v

    return Vec(
        cos_rz * (cos_ry * v[0] - sin_ry * (-sin_rx * v[1] + cos_rx * v[2])) + sin_rz * (cos_rx * v[1] + sin_rx * v[2]),
        -sin_rz * (cos_ry * v[0] - sin_ry * (-sin_rx * v[1] + cos_rx * v[2])) + cos_rz * (cos_rx * v[1] + sin_rx * v[2]),
        sin_ry * v[0] + cos_ry * (-sin_rx * v[1] + cos_rx * v[2])
    )

def check_collision(pos, v, objs):
    min_d = None
    min_p = None
    min_o = None

    for obj in objs:
        (d, p) = obj.collision(pos, v)
        if (d is not None and (min_d == None or d < min_d)):
            min_d = d
            min_p = p
            min_o = obj

    return (min_d, min_p, min_o)

clamp = lambda x, m, M: max(min(x, M), m)

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

def cast_ray(pos, d, scene, depth=0):
    hit_clr = scene.background_color

    if (depth > DEPTH_MAX):
        return hit_clr

    (min_d, min_p, min_o) = check_collision(pos, d, scene.objs)

    if (min_d != None):
        material = scene.materials[min_o.material]

        n = min_o.normal(min_p)

        # TODO reflection is broken, refraction is not
        if (material.refractive and material.reflective):
            reflection_dir = ~reflect(d, n)
            refraction_dir = ~refract(d, n, material.ior)
            reflection_orig = min_p - n * scene.bias if (reflection_dir.dot(n) < 0) else min_p + n * scene.bias
            refraction_orig = min_p - n * scene.bias if (refraction_dir.dot(n) < 0) else min_p + n * scene.bias
            reflection_clr = cast_ray(reflection_orig, reflection_dir, scene, depth=depth+1)
            refraction_clr = cast_ray(refraction_orig, refraction_dir, scene, depth=depth+1)
            kr = fresnel(d, n, material.ior)
            hit_clr = reflection_clr * kr + refraction_clr * (1 - kr)
        elif (material.reflective):
            kr = fresnel(d, n, material.ior)
            reflection_dir = reflect(d, n)
            reflection_orig = min_p + n * scene.bias if (reflection_dir.dot(n) < 0) else min_p - n * scene.bias
            hit_clr = cast_ray(reflection_orig, reflection_dir, scene, depth=depth+1) * kr
        else:
            light_amt = 0
            spec_clr = Vec(0.0, 0.0, 0.0)
            shadow_orig = min_p + n * scene.bias if (d ** n < 0) else min_p - n * scene.bias

            for light in scene.lights:
                vec = min_p.to(light.pos)
                ld2 = vec.mag2()
                vec = ~vec
                ldn = max(vec ** n, 0)
                (s_d, s_p, s_o) = check_collision(shadow_orig, vec, scene.objs)
                if (s_d == None or s_d >= ld2):
                    light_amt += light.intensity * ldn
                reflection_dir = (-vec).reflect(n)
                spec_clr += pow(max(-(reflection_dir ** d), 0), material.spec_exponent) * light.intensity

            #hit_clr = light_amt * min_o.eval_diffuse_color(st) * material.Kd + spec_clr * material.Ks
            hit_clr = material.diffuse_color * light_amt * material.Kd + spec_clr * material.Ks

    return hit_clr

def render_scene_chunk(scene, camera, rs, re, n, q):
    t = time.time()
    print('Started render thread {}'.format(n))

    screen = [[(0, 0, 0) for j in range(camera.res[0])] for i in range(re - rs)]

    cos_rx, cos_ry, cos_rz = math.cos(camera.rot[0]), math.cos(camera.rot[1]), math.cos(camera.rot[2])
    sin_rx, sin_ry, sin_rz = math.sin(camera.rot[0]), math.sin(camera.rot[1]), math.sin(camera.rot[2])

    half_width = -math.tan(math.radians(camera.fov[0] / 2.0))
    half_height = -math.tan(math.radians(camera.fov[1] / 2.0))

    for y in range(rs, re):
        for x in range(camera.res[0]):
            vec = ~rot(Vec(
               -half_width * (x / (camera.res[0] / 2) - 1),
               half_height * (y / (camera.res[1] / 2) - 1),
               -1.0
            ), cos_rx, cos_ry, cos_rz, sin_rx, sin_ry, sin_rz)

            c = cast_ray(camera.pos, vec, scene)
            c = (int(0xff * c[0]), int(0xff * c[1]), int(0xff * c[2]))
            screen[y - rs][x] = c

    q.put((n, screen))

    u = time.time()
    print('Render thread {} finished in {} seconds'.format(n, u - t))

def render_scene_multithreaded(scene, camera, job_count=4):
    multiprocessing.freeze_support()
    t = time.time()
    q = multiprocessing.Queue(job_count)

    for n in range(job_count):
        rs = camera.res[1] // job_count * n
        re = camera.res[1] // job_count * (n + 1)

        multiprocessing.Process(
            target = render_scene_chunk,
            args = (scene, camera, rs, re, n, q)
        ).start()

    slices = [q.get() for n in range(job_count)]
    screen = [r for s in sorted(slices, key=lambda i: i[0]) for r in s[1]]

    u = time.time()
    print("Rendered scene in {} seconds".format(u - t))

    return screen

def image_from_pixels(pixels):
    dim = (len(pixels[0]), len(pixels))
    img = Image.new('RGB', dim, 'black')
    pix = img.load()
    for i in range(dim[0]):
        for j in range(dim[1]):
            pix[i, j] = pixels[j][i]
    return img

############
### MAIN ###
############

scene = Scene(
    background_color = Vec(0.235294, 0.67451, 0.843137),
    bias = 0.00001,
    materials = {
        'glass': Material(
            reflective = True,
            refractive = True,
            spec_exponent = 25,
            diffuse_color = Vec(0.2, 0.2, 0.2),
            Kd = 0.8,
            Ks = 0.2,
            ior = 1.5
        ),
        'glossy': Material(
            reflective = False,
            refractive = False,
            spec_exponent = 25,
            diffuse_color = Vec(0.8, 0.7, 0.2),
            Kd = 0.8,
            Ks = 0.2,
            ior = 1.3
        ),
        'rubber': Material(
            reflective = False,
            refractive = False,
            spec_exponent = 25,
            diffuse_color = Vec(0.2, 0.2, 0.2),
            Kd = 0.8,
            Ks = 0.2,
            ior = 1.3
        )
    },
    objs = [
        Sphere(
            pos = Vec(-1, 0, -12),
            radius = 2.0,
            material = 'glossy'
        ),
        Sphere(
            pos = Vec(0.5, -0.5, -8),
            radius = 1.5,
            material = 'glass'
        ),
        Sphere(
            pos = Vec(1.5, -3, -8),
            radius = 1.0,
            material = 'rubber'
        )
    ],
    lights = [
        Light(
            pos = Vec(-20, 70, 20),
            intensity = 0.5
        ),
        Light(
            pos = Vec(30, 50, -12),
            intensity = 1.0
        )
    ]
)

camera = Camera(
    fov = (90, 50.625),
    res = (3840, 2160),
    pos = Vec(0, 0, 0),
    rot = Vec(0, 0, 0)
)

if __name__ == '__main__':
    screen = render_scene_multithreaded(scene, camera)
    img = image_from_pixels(screen)
    img.save('big.bmp')
    img.show()