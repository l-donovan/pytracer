from tracelib import *

scene = Scene(
    background_color=STD_COLORS['celestial-blue'],
    materials=STD_MATERIALS,
    # x is horizontal position (right-positive)
    # y is vertical position (top-positive)
    # z is distance from camera (far-positive)
    objs=[
        Sphere(
            pos=Vec(-1.0, 1.0, 7.0),
            radius=2.0,
            material='glass'
        ),
        Sphere(
            pos=Vec(-0.75, -1.0, 12.0),
            radius=2.0,
            material='glass'
        ),
        Sphere(
            pos=Vec(3.0, 0.5, 6.0),
            radius=1.5,
            material='glass'
        ),
        Sphere(
            pos=Vec(2.5, 0.0, 4.0),
            radius=1.0,
            material='glass'
        ),
        Plane(
            v0=Vec(0, -5, 0),
            v1=Vec(1, -5, 0),
            v2=Vec(1, -5, 1),
            material='floor'
        )
    ],
    lights=[
        Light(
            pos=Vec(-20, 70, -20),
            intensity=0.5
        ),
        Light(
            pos=Vec(30, 50, 12),
            intensity=1.0
        ),
        Light(
            pos=Vec(10, -30, 12),
            intensity=0.4
        )
    ]
)

camera = Camera(
    fov=90.0,
    res=(960, 540),
    pos=Vec(0, 0, 0),
    rot=Vec(0, 0, 0)
)

options = Options(
    max_depth=5,
    proc_count=4
)

if __name__ == '__main__':
    screen = render_scene(scene, camera, options)
    img = image_from_pixels(screen)
    img.save('render.bmp')
    img.show()
