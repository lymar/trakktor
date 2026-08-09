"""Render the photographed-page corpus in Blender.

The corpus exists to answer one question the flat renders cannot: what a
photograph does to a page that a scan does not. So every frame here is the
*same* typeset page, seen through one deliberate defect — a camera off the
normal, a sheet bent along its spine, a hand between the lamp and the paper —
and the flat 300 dpi render of the same PDF is the reference every frame is
measured against.

Two properties are worth stating because the whole measurement rests on them:

* **The text is never stretched.** A bent sheet is built by walking arc length
  along the bend, not by displacing a flat grid, so a word near the gutter
  keeps the width it has on the flat page. Paper does not stretch, and a
  corpus whose paper stretches would measure the wrong thing.
* **Every frame is reproducible.** Positions, angles, light energies and the
  sampling seed are constants in this file; nothing is randomized per run.

Usage:

    blender -b -P render.py -- --texture <page.png> --spread <spread.png> \\
        --out <dir> [--scene <name> ...] [--samples 128]
"""

import argparse
import math
import os
import sys

import bpy
from mathutils import Quaternion, Vector

# ---------------------------------------------------------------------------
# Geometry constants. The page is 5.5 x 8.5 inches, in metres, because Blender
# thinks in metres and a lamp two metres away has to be two metres away for
# its falloff to look right.
# ---------------------------------------------------------------------------

INCH = 0.0254
PAGE_W = 5.5 * INCH
PAGE_H = 8.5 * INCH

# Subdivisions across a sheet. Enough that a bend reads as a curve and not as
# a fan of facets; the renderer smooths the normals on top of this.
GRID_U = 120
GRID_V = 160

# Phone-camera geometry: a 26 mm-equivalent lens on a 36 mm frame.
FOCAL_MM = 26.0
SENSOR_MM = 36.0

# The frame. A page shot to fill most of it lands near the 300 dpi reference
# in pixels per line, which is what keeps this a test of geometry and light
# rather than a test of resolution.
PORTRAIT = (2304, 3072)
LANDSCAPE = (3072, 2304)


# ---------------------------------------------------------------------------
# Scene plumbing
# ---------------------------------------------------------------------------


def reset() -> None:
    """Empty the file, including the data that survives object deletion."""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def configure(samples: int, resolution: tuple[int, int],
              percent: int = 100) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.seed = 0
    scene.cycles.use_animated_seed = False
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.resolution_percentage = percent
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False

    # A photograph of white paper is white paper. AgX would grey it down in a
    # way no phone does, and the greying would show up in the measurement as
    # if the geometry had caused it.
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"

    prefs = bpy.context.preferences.addons.get("cycles")
    if prefs is not None:
        cprefs = prefs.preferences
        try:
            cprefs.compute_device_type = "METAL"
            cprefs.get_devices()
            # The GPU alone, not the GPU and the CPU together: mixing them
            # splits the frame between two samplers and the seam shows.
            for device in cprefs.devices:
                device.use = device.type == "METAL"
            scene.cycles.device = "GPU"
        except (TypeError, AttributeError):
            scene.cycles.device = "CPU"


def world(strength: float = 0.22, color=(0.55, 0.58, 0.62)) -> None:
    """Ambient fill: an overcast room, not a black void."""
    wd = bpy.data.worlds.new("World")
    wd.use_nodes = True
    bg = wd.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (*color, 1.0)
    bg.inputs["Strength"].default_value = strength
    bpy.context.scene.world = wd


def set_input(node, name: str, value) -> None:
    """Set a shader input if this Blender has it, and say nothing if not."""
    if name in node.inputs:
        node.inputs[name].default_value = value


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------


def paper_material(texture: str, gloss: bool = False):
    mat = bpy.data.materials.new("Paper")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]

    image = nodes.new("ShaderNodeTexImage")
    image.image = bpy.data.images.load(texture)
    image.image.colorspace_settings.name = "sRGB"
    image.interpolation = "Cubic"
    image.extension = "EXTEND"
    links.new(image.outputs["Color"], bsdf.inputs["Base Color"])

    # Uncoated book paper is rough and barely specular; the glossy variant is
    # a coated stock, which is what makes a lamp leave a highlight.
    set_input(bsdf, "Roughness", 0.20 if gloss else 0.62)
    set_input(bsdf, "Specular IOR Level", 0.95 if gloss else 0.22)
    set_input(bsdf, "IOR", 1.45)
    set_input(bsdf, "Sheen Weight", 0.06)
    set_input(bsdf, "Metallic", 0.0)
    return mat


def desk_material():
    mat = bpy.data.materials.new("Desk")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]

    coords = nodes.new("ShaderNodeTexCoord")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 9.0
    noise.inputs["Detail"].default_value = 6.0
    # Grain runs along one axis, the way sawn wood does; without the stretch
    # the desk reads as camouflage rather than as a table.
    stretch = nodes.new("ShaderNodeMapping")
    stretch.inputs["Scale"].default_value = (34.0, 1.4, 1.0)
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.075, 0.042, 0.022, 1.0)
    ramp.color_ramp.elements[1].color = (0.185, 0.110, 0.058, 1.0)

    links.new(coords.outputs["Object"], stretch.inputs["Vector"])
    links.new(stretch.outputs["Vector"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    set_input(bsdf, "Roughness", 0.45)
    set_input(bsdf, "Specular IOR Level", 0.3)
    return mat


def matte_material(name: str, color, roughness: float = 0.7):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    set_input(bsdf, "Base Color", (*color, 1.0))
    set_input(bsdf, "Roughness", roughness)
    set_input(bsdf, "Specular IOR Level", 0.25)
    return mat


# ---------------------------------------------------------------------------
# The sheet
# ---------------------------------------------------------------------------


def bend_profile(u: float, kind: str, width: float):
    """Where a point at arc length `u * width` from the left edge lands.

    Returns `(x, z)` measured from that left edge. The parameter is arc
    length, so the distance between two points along the surface is the
    distance between them on the flat page — the sheet bends, it does not
    stretch, and a bent sheet is therefore narrower than a flat one.
    """
    if kind == "flat":
        return u * width, 0.0

    # Tangent angle as a function of arc length, integrated below. Each shape
    # is a slope that starts somewhere and decays; the integral is the curve.
    def slope(s: float) -> float:
        t = s / width
        if kind == "curl":
            # One edge lifted off the table and relaxing back down: steep at
            # the left, flat by the middle, and flat from there on.
            return math.radians(52.0) * math.exp(-5.0 * t)
        if kind == "half":
            # Half of a spread, gutter at u = 0: the paper leaves the gutter
            # climbing, crests about a third of the way out, and settles onto
            # the block below it.
            return math.radians(58.0) * (1.0 - 3.2 * t) * math.exp(-4.0 * t)
        raise ValueError(f"unknown bend {kind!r}")

    steps = 512
    ds = width / steps
    x = z = 0.0
    target = u * width
    walked = 0.0
    while walked + ds <= target:
        theta = slope(walked + ds / 2.0)
        x += math.cos(theta) * ds
        z += math.sin(theta) * ds
        walked += ds
    rest = target - walked
    if rest > 0.0:
        theta = slope(walked + rest / 2.0)
        x += math.cos(theta) * rest
        z += math.sin(theta) * rest
    return x, z


def make_sheet(
    name: str,
    material,
    width: float = PAGE_W,
    height: float = PAGE_H,
    bend: str = "flat",
    mirror: bool = False,
    uv_u: tuple[float, float] = (0.0, 1.0),
    center: bool = True,
):
    """A textured sheet, optionally bent, UV-mapped by arc length.

    `uv_u` is the slice of the texture this sheet carries, which is how the
    two halves of a spread share one image. `center` is off for a spread
    half, whose origin belongs at the gutter and not at its own middle.
    """
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    verts, faces = [], []
    profile = [bend_profile(i / GRID_U, bend, width) for i in range(GRID_U + 1)]
    if center:
        span = profile[-1][0]
        profile = [(x - span / 2.0, z) for x, z in profile]
    if mirror:
        profile = [(-x, z) for x, z in profile]

    for j in range(GRID_V + 1):
        y = (j / GRID_V - 0.5) * height
        for i in range(GRID_U + 1):
            x, z = profile[i]
            verts.append((x, y, z))

    stride = GRID_U + 1
    for j in range(GRID_V):
        for i in range(GRID_U):
            a = j * stride + i
            faces.append((a, a + 1, a + stride + 1, a + stride))

    mesh.from_pydata(verts, [], faces)
    mesh.update()

    layer = mesh.uv_layers.new(name="UVMap")
    u0, u1 = uv_u
    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            vi = mesh.loops[loop_index].vertex_index
            i, j = vi % stride, vi // stride
            u = i / GRID_U
            if mirror:
                u = 1.0 - u
            layer.uv[loop_index].vector = (u0 + u * (u1 - u0), j / GRID_V)

    for poly in mesh.polygons:
        poly.use_smooth = True
    obj.data.materials.append(material)
    return obj


def make_plane(name: str, size: float, material, location=(0, 0, 0)):
    bpy.ops.mesh.primitive_plane_add(size=size, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    return obj


def make_book_block(half_width: float, height: float, top: float):
    """The closed pages under a spread: two slabs meeting at the gutter."""
    block = matte_material("Block", (0.80, 0.77, 0.70), roughness=0.85)
    for sign in (-1.0, 1.0):
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        cube = bpy.context.object
        cube.name = f"Block{'L' if sign < 0 else 'R'}"
        cube.scale = (half_width * 0.99, height * 1.015, top)
        cube.location = (sign * half_width / 2.0 * 0.99, 0.0, top / 2.0)
        cube.data.materials.append(block)


# ---------------------------------------------------------------------------
# Camera and light
# ---------------------------------------------------------------------------


def make_camera(location, target=(0.0, 0.0, 0.0), roll: float = 0.0):
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = FOCAL_MM
    cam_data.sensor_width = SENSOR_MM
    # A page is photographed from twenty centimetres; the default near plane
    # is at ten, and a hand between lens and paper would vanish through it.
    cam_data.clip_start = 0.004
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    cam.location = Vector(location)
    aim = (Vector(target) - Vector(location)).to_track_quat("-Z", "Y")
    # Roll is about the camera's own view axis, which is what holding a phone
    # a few degrees off level does to the frame — it turns the picture, not
    # the page.
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = aim @ Quaternion((0.0, 0.0, 1.0), roll)
    return cam


def area_light(name, location, target, size, irradiance,
               color=(1.0, 0.97, 0.92)):
    """A lamp specified by what it does to the page, not by its wattage.

    Cycles takes a light's radiant power, so a lamp moved closer has to be
    dimmed by hand or the page blows out. Every scene here wants the same
    exposure on the paper and differs only in where the light comes from, so
    the scenes state the irradiance they want and the power follows from the
    distance: a Lambertian emitter of power `P` puts `P / (pi * r^2)` on a
    surface facing it.
    """
    location, target = Vector(location), Vector(target)
    radius = (target - location).length
    data = bpy.data.lights.new(name, type="AREA")
    data.energy = irradiance * math.pi * radius * radius
    data.size = size
    data.color = color
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    obj.rotation_euler = (target - location).to_track_quat("-Z", "Y").to_euler()
    return obj


# White paper under this much light lands just below clipping, which is where
# a phone's auto-exposure puts it. Every scene is lit to the same total.
PAPER_IRRADIANCE = 3.05
WORLD_SHARE = 0.13


def sun_direction(elevation_deg: float, azimuth_deg: float) -> Vector:
    """The direction sunlight travels, as a unit vector."""
    elevation, azimuth = math.radians(elevation_deg), math.radians(azimuth_deg)
    return Vector(
        (
            -math.cos(elevation) * math.sin(azimuth),
            -math.cos(elevation) * math.cos(azimuth),
            -math.sin(elevation),
        )
    )


def shadow_caster(target, height: float, direction: Vector) -> Vector:
    """Where to hold an object so its shadow lands on `target`."""
    travel = direction * (height / -direction.z)
    return Vector((target[0] - travel.x, target[1] - travel.y, height))


def sun_light(name, irradiance, elevation_deg, azimuth_deg, angle_deg=0.6,
              color=(1.0, 0.96, 0.90)):
    """A far light, for when a shadow has to keep its edges.

    A lamp close enough to throw a hand's shadow onto a page also throws a
    penumbra several centimetres wide, and the shadow stops looking like a
    hand. Sunlight through a window is parallel, so the shadow is as sharp as
    the source is small however far the hand is from the paper — which is what
    lets the hand itself stay outside the frame.
    """
    data = bpy.data.lights.new(name, type="SUN")
    data.energy = irradiance
    data.angle = math.radians(angle_deg)
    data.color = color
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    direction = sun_direction(elevation_deg, azimuth_deg)
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return obj


def mirror_light(name, camera, point, distance, size, irradiance, color):
    """A lamp placed where the page will reflect it straight into the lens.

    A highlight is a picture of the lamp, so it cannot be put on the page by
    hand: the lamp goes on the mirrored camera ray and the reflection lands
    where it lands.
    """
    camera, point = Vector(camera), Vector(point)
    view = camera - point
    mirrored = Vector((-view.x, -view.y, view.z)).normalized()
    return area_light(name, point + mirrored * distance, point, size,
                      irradiance, color)


def make_hand(location, rotation, scale: float = 1.0):
    """A blocker shaped enough like a hand to cast a hand's shadow."""
    skin = matte_material("Skin", (0.55, 0.36, 0.27), roughness=0.6)
    parts = []

    bpy.ops.mesh.primitive_cube_add(size=1.0)
    palm = bpy.context.object
    palm.name = "Palm"
    palm.scale = (0.052, 0.070, 0.011)
    bpy.ops.object.modifier_add(type="BEVEL")
    palm.modifiers["Bevel"].width = 0.012
    palm.modifiers["Bevel"].segments = 6
    parts.append(palm)

    for k, (dx, length, lean) in enumerate(
        [
            (-0.036, 0.072, 0.10),
            (-0.012, 0.082, 0.03),
            (0.012, 0.078, -0.03),
            (0.034, 0.066, -0.10),
        ]
    ):
        bpy.ops.mesh.primitive_cylinder_add(radius=0.0092, depth=length)
        finger = bpy.context.object
        finger.name = f"Finger{k}"
        finger.rotation_euler = (math.radians(90.0), 0.0, lean)
        finger.location = (dx, 0.070 + length / 2.0 * 0.92, 0.0)
        finger.scale = (1.0, 1.0, 0.62)
        parts.append(finger)

    # The thumb, which is what makes the shadow read as a hand.
    bpy.ops.mesh.primitive_cylinder_add(radius=0.0115, depth=0.062)
    thumb = bpy.context.object
    thumb.name = "Thumb"
    thumb.rotation_euler = (math.radians(90.0), 0.0, math.radians(58.0))
    thumb.location = (-0.058, 0.040, 0.0)
    thumb.scale = (1.0, 1.0, 0.62)
    parts.append(thumb)

    for part in parts:
        part.data.materials.append(skin)

    # Bake each part's own scale into its mesh before joining. Without this
    # the joined object inherits the palm's scale as its object scale, and the
    # `scale` asked for here would overwrite it — a hand a hundred times life
    # size, which is a black frame and not an obvious mistake.
    bpy.ops.object.select_all(action="DESELECT")
    for part in parts:
        part.select_set(True)
    bpy.context.view_layer.objects.active = palm
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    bpy.ops.object.join()

    hand = bpy.context.object
    hand.name = "Hand"
    hand.location = Vector(location)
    hand.rotation_euler = rotation
    hand.scale = (scale, scale, scale)
    return hand


def make_clutter():
    """The desk a page is actually photographed on."""
    pen = matte_material("Pen", (0.06, 0.07, 0.09), roughness=0.35)
    mug = matte_material("Mug", (0.72, 0.70, 0.66), roughness=0.5)
    pad = matte_material("Pad", (0.10, 0.32, 0.36), roughness=0.8)

    bpy.ops.mesh.primitive_cylinder_add(radius=0.0055, depth=0.145)
    obj = bpy.context.object
    obj.rotation_euler = (math.radians(90.0), 0.0, math.radians(24.0))
    obj.location = (0.135, -0.055, 0.0055)
    obj.data.materials.append(pen)

    bpy.ops.mesh.primitive_cylinder_add(radius=0.041, depth=0.095)
    obj = bpy.context.object
    obj.location = (-0.165, 0.115, 0.0475)
    obj.data.materials.append(mug)

    bpy.ops.mesh.primitive_cube_add(size=1.0)
    obj = bpy.context.object
    obj.scale = (0.115, 0.082, 0.0035)
    obj.location = (0.175, 0.135, 0.0035)
    obj.rotation_euler = (0.0, 0.0, math.radians(-9.0))
    obj.data.materials.append(pad)


# ---------------------------------------------------------------------------
# The scenes
# ---------------------------------------------------------------------------


def base_scene(texture: str, gloss: bool = False, ambient: float = WORLD_SHARE):
    # The ambient term is stated as a share of the page's total exposure, so
    # that dimming the room to deepen a shadow does not also darken the page.
    world(strength=PAPER_IRRADIANCE * ambient / (math.pi * 0.58))
    make_plane("Desk", 3.0, desk_material())
    return paper_material(texture, gloss=gloss)


def key_light(share: float = 0.67, location=(-0.30, -0.22, 0.55), size=0.40):
    return area_light("Key", location, (0.0, 0.0, 0.0), size,
                      PAPER_IRRADIANCE * share)


def fill_light(share: float = 0.20, location=(0.42, 0.34, 0.42)):
    return area_light("Fill", location, (0, 0, 0), 0.7,
                      PAPER_IRRADIANCE * share)


def frame_distance(subject: float, fill: float) -> float:
    """How far back to stand so `subject` metres fill `fill` of the frame.

    Blender fits the sensor to the longer side of the frame, and every frame
    here is composed along its longer side, so one formula serves both
    orientations.
    """
    return subject / (fill * 2.0 * math.tan(math.atan(SENSOR_MM / 2.0 / FOCAL_MM)))


def camera_over(fill: float, tilt_deg: float, azimuth_deg: float = 0.0,
                roll_deg: float = 0.0, target=(0.0, 0.0, 0.0),
                subject: float = PAGE_H):
    """A camera `tilt_deg` off straight down, framing the page to `fill`.

    Distance follows from the framing rather than the other way round: a
    photographer moves until the page fills the screen, and the corpus is
    only fair if every frame is composed the same way.
    """
    distance = frame_distance(subject, fill)
    tilt = math.radians(tilt_deg)
    azimuth = math.radians(azimuth_deg)
    offset = Vector(
        (
            math.sin(tilt) * math.sin(azimuth),
            -math.sin(tilt) * math.cos(azimuth),
            math.cos(tilt),
        )
    ) * distance
    return make_camera(Vector(target) + offset, target, math.radians(roll_deg))


def scene_overhead(texture, **_):
    mat = base_scene(texture)
    sheet = make_sheet("Page", mat)
    sheet.location.z = 0.0006
    key_light()
    fill_light()
    camera_over(0.90, 4.0)
    return PORTRAIT


def scene_angle(texture, tilt, azimuth=0.0, roll=0.0, fill=0.88, **_):
    mat = base_scene(texture)
    sheet = make_sheet("Page", mat)
    sheet.location.z = 0.0006
    key_light()
    fill_light()
    camera_over(fill, tilt, azimuth, roll)
    return PORTRAIT


def scene_rotated(texture, degrees, **_):
    mat = base_scene(texture)
    sheet = make_sheet("Page", mat)
    sheet.location.z = 0.0006
    sheet.rotation_euler = (0.0, 0.0, math.radians(degrees))
    key_light()
    fill_light()
    camera_over(0.88, 5.0)
    return LANDSCAPE if degrees % 180 == 90 else PORTRAIT


def scene_curl(texture, **_):
    mat = base_scene(texture)
    make_sheet("Page", mat, bend="curl")
    key_light(location=(-0.34, -0.18, 0.50))
    fill_light()
    camera_over(0.84, 22.0, azimuth_deg=8.0)
    return PORTRAIT


def scene_spread(texture, **_):
    """A book open at pages 42 and 43, gutter down the middle."""
    mat = base_scene(texture)
    block_top = 0.017
    make_book_block(PAGE_W * 2.0, PAGE_H, block_top)

    left = make_sheet("PageL", mat, bend="half", mirror=True, center=False,
                      uv_u=(0.0, 0.5))
    left.location = (0.0, 0.0, block_top)
    right = make_sheet("PageR", mat, bend="half", center=False,
                       uv_u=(0.5, 1.0))
    right.location = (0.0, 0.0, block_top)

    key_light(location=(-0.22, -0.34, 0.58))
    fill_light(location=(0.34, 0.30, 0.40))
    camera_over(0.86, 11.0, azimuth_deg=6.0, target=(0.0, 0.0, block_top),
                subject=PAGE_W * 1.86)
    return LANDSCAPE


def scene_shadow(texture, **_):
    # A window rather than a lamp, and a dim room behind it: a broad source
    # would wash the shadow away and leave nothing to measure.
    mat = base_scene(texture, ambient=0.11)
    sheet = make_sheet("Page", mat)
    sheet.location.z = 0.0006
    elevation, azimuth = 54.0, -38.0
    sun_light("Sun", PAPER_IRRADIANCE * 0.80, elevation, azimuth)
    fill_light(share=0.09, location=(0.40, 0.30, 0.30))

    # High enough above the page to fall outside the frame, and placed so that
    # the sun throws its shadow across the text rather than beside it.
    make_hand(
        shadow_caster((-0.020, -0.040), 0.132,
                      sun_direction(elevation, azimuth)),
        (0.0, 0.0, math.radians(-78.0)),
        scale=0.78,
    )
    camera_over(0.90, 7.0)
    return PORTRAIT


def scene_glare(texture, **_):
    mat = base_scene(texture, gloss=True, ambient=0.10)
    sheet = make_sheet("Page", mat)
    sheet.location.z = 0.0006
    camera = camera_over(0.86, 18.0)
    mirror_light("Lamp", camera.location, (0.010, 0.052, 0.0), 0.26, 0.075,
                 PAPER_IRRADIANCE * 0.78, (1.0, 0.93, 0.80))
    fill_light(share=0.16, location=(-0.36, -0.30, 0.36))
    return PORTRAIT


def scene_finger(texture, **_):
    mat = base_scene(texture)
    make_sheet("Page", mat, bend="curl")
    make_hand((-0.052, -0.135, 0.0035), (0.0, 0.0, math.radians(-18.0)),
              scale=0.80)
    key_light(location=(0.26, -0.26, 0.52))
    fill_light()
    camera_over(0.82, 26.0, azimuth_deg=-6.0)
    return PORTRAIT


def scene_desk(texture, **_):
    mat = base_scene(texture)
    sheet = make_sheet("Page", mat)
    sheet.location = (0.012, -0.004, 0.0006)
    sheet.rotation_euler = (0.0, 0.0, math.radians(-6.0))
    make_clutter()
    key_light()
    fill_light()
    camera_over(0.60, 24.0, azimuth_deg=-14.0)
    return PORTRAIT


SCENES = {
    "overhead": (scene_overhead, "page", {}),
    "angle-20": (scene_angle, "page", {"tilt": 20.0}),
    "angle-40": (scene_angle, "page", {"tilt": 40.0, "azimuth": 12.0}),
    "tilt-12": (scene_angle, "page", {"tilt": 6.0, "roll": 12.0}),
    "rot-90": (scene_rotated, "page", {"degrees": 90}),
    "rot-180": (scene_rotated, "page", {"degrees": 180}),
    "curl": (scene_curl, "page", {}),
    "spread": (scene_spread, "spread", {}),
    "shadow-hand": (scene_shadow, "page", {}),
    "glare": (scene_glare, "page", {}),
    "finger": (scene_finger, "page", {}),
    "desk": (scene_desk, "page", {}),
}


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--texture", required=True, help="the single page")
    parser.add_argument("--spread", required=True, help="the two-page spread")
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene", action="append", default=None)
    parser.add_argument("--samples", type=int, default=160)
    parser.add_argument("--percent", type=int, default=100,
                        help="render at a fraction of the size, for setup")
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    wanted = args.scene or list(SCENES)
    textures = {"page": args.texture, "spread": args.spread}

    for name in wanted:
        build, which, kwargs = SCENES[name]
        reset()
        resolution = build(textures[which], **kwargs)
        configure(args.samples, resolution, args.percent)
        bpy.context.scene.render.filepath = os.path.join(args.out, name)
        print(f"[corpus] rendering {name} at {resolution[0]}x{resolution[1]}")
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
