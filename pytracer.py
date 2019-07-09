from veclib import Vec
from dataclasses import dataclass
import math, time

import multiprocessing

from PIL import Image

REFLECTION_LIMIT = 4

@dataclass
class Camera:
    fov: list # hfov, vfov
    res: list # w, h
    pos: Vec  # x, y, z
    rot: Vec  # roll, pan, tilt
    dscreen: float

@dataclass
class Scene:
    materials: dict
    objs: list
    lights: list

class Material:
    pass

class Light:
    def __init__(self, pos):
        self.pos = pos

    def collision(self, p, d):
        m = p - self.pos
        b = m ** d
        c = m.mag2() - 0.5

        if (c > 0 and b > 0):
            return (None, None)

        discr = b ** 2 - c

        if (discr < 0):
            return (None, None)

        t = max(-b - math.sqrt(discr), 0)
        q = p + d * t

        return (t, q)

class Sphere:
    def __init__(self, pos, radius, material):
        self.pos = pos
        self.r2 = radius ** 2
        self.material = material

    def collision(self, p, d):
        m = p - self.pos
        b = m ** d
        c = m.mag2() - self.r2

        if (c > 0 and b > 0):
            return (None, None)

        discr = b ** 2 - c

        if (discr < 0):
            return (None, None)

        t = max(-b - math.sqrt(discr), 0)
        q = p + d * t

        return (t, q)

    def reflection(self, p, d, q):
        n = ~(q - self.pos)
        return d.reflect(n)

def rot(v, cos_rx, cos_ry, cos_rz, sin_rx, sin_ry, sin_rz):
    # `rot` uses cached trig values for a nice bump in efficiency

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

def get_color(pos, light, objs, second_chance=False):
    vec = ~(pos.to(light.pos))

    (min_d, min_p, min_o) = check_collision(pos, vec, objs)

    if (min_d != None):
        # Accounts for floating point errors
        if (min_d < 0.5 and not second_chance):
            return get_color(pos + vec * .01, light, objs, second_chance=True)
        else:
            return 0
    else:
        return 1.0 / ((light.pos.dist(pos) * 0.1 + 1) ** 2)

def render_scene(scene, camera, rs, re, n, q):
    t = time.time()
    print('Started render thread {}'.format(n))

    screen = [[(0, 0, 0) for j in range(camera.res[0])] for i in range(re - rs)]

    cos_rx, cos_ry, cos_rz = math.cos(camera.rot[0]), math.cos(camera.rot[1]), math.cos(camera.rot[2])
    sin_rx, sin_ry, sin_rz = math.sin(camera.rot[0]), math.sin(camera.rot[1]), math.sin(camera.rot[2])

    half_width = camera.dscreen * math.tan(math.radians(camera.fov[0] / 2.0))
    half_height = camera.dscreen * math.tan(math.radians(camera.fov[1] / 2.0))

    for y in range(rs, re):
        for x in range(camera.res[0]):
            vec = ~rot(Vec(
               half_width * (x / (camera.res[0] / 2) - 1),
               half_height * (y / (camera.res[1] / 2) - 1),
               camera.dscreen
            ), cos_rx, cos_ry, cos_rz, sin_rx, sin_ry, sin_rz)

            (min_d, min_p, min_o) = check_collision(camera.pos, vec, scene.objs)

            if (min_d != None):
                max_c = 0
                for light in scene.lights:
                    c = get_color(min_p, light, scene.objs)
                    if (c > max_c):
                        max_c = c
                screen[y - rs][x] = (int(0xff * max_c), 0, 0)

            (min_d, min_p, min_o) = check_collision(camera.pos, vec, scene.lights)

            if (min_d != None):
                screen[y - rs][x] = (0xff, 0xff, 0xff)

    q.put((n, screen))

    u = time.time()
    print('Render thread {} finished in {} seconds'.format(n, u - t))

############
### MAIN ###
############

scene = Scene(
    materials = {
        'glass': Material(

        )
    },
    objs = [
        Sphere(
            pos = Vec(20, 0, 30),
            radius = 5.0,
            material = 'glass'
        ),
        Sphere(
            pos = Vec(20, -8, 30),
            radius = 2.0,
            material = 'glass'
        ),
        Sphere(
            pos = Vec(-5, -13, 30),
            radius = 2.0,
            material = 'glass'
        )
    ],
    lights = [
        Light(pos = Vec(0, -5.5, 30))
    ]
)

camera = Camera(
    fov = [90, 50.625],
    res = [960, 580],
    pos = Vec(0, 0, 0),
    rot = Vec(0, 0, 0),
    dscreen = 10.0
)

if __name__ == '__main__':
    multiprocessing.freeze_support()

    t = time.time()

    job_count = 4

    q = multiprocessing.Queue(job_count)

    for n in range(job_count):
        rs = camera.res[1] // job_count * n
        re = camera.res[1] // job_count * (n + 1)

        multiprocessing.Process(
            target = render_scene,
            args = (scene, camera, rs, re, n, q)
        ).start()

    slices = [q.get() for n in range(job_count)]
    screen = [r for s in sorted(slices, key=lambda i: i[0]) for r in s[1]]

    img = Image.new('RGB', (camera.res[0], camera.res[1]), 'black')
    pixels = img.load()
    for i in range(camera.res[0]):
        for j in range(camera.res[1]):
            pixels[i, j] = screen[j][i]
    img.show()

    u = time.time()
    print("Rendered scene in {} seconds".format(u - t))