"""Based on lorenz.py in glumpy repo."""
from __future__ import division
from glumpy import app
from glumpy.transforms import Position, Trackball, Viewport
from grid import grid
from skeleton_vis import LimbCollection


window = app.Window(width=1000, height=800, color=(1, 1, 1, 1))
transform = Trackball(Position())
viewport = Viewport()

pose_scale = 2.5
ticks = grid(transform, viewport, scale=pose_scale)
limb_collections = []


@window.event
def on_draw(dt):
    window.clear()
    ticks.draw()
    for limb_collection in limb_collections:
        limb_collection.draw()


@window.event
def on_key_press(key, modifiers):
    if key == app.window.key.SPACE:
        reset()


def reset():
    transform.theta = 45
    transform.phi = 0
    transform.zoom = 16.5


reset()
for s in ['transform', 'viewport']:
    window.attach(ticks[s])


def add_limb_collection(skeleton, pose, linewidth=2.0):
    limb_collection = LimbCollection(
        skeleton, pose / pose_scale, transform, viewport, linewidth=linewidth)
    limb_collections.append(limb_collection)
    return limb_collection


def run(fps=60, duration=None):
    kwargs = {
        'framerate': fps
    }
    if duration is not None:
        kwargs['duration'] = duration
    app.run(**kwargs)
    # app.run(framerate=fps, duration=duration)


def save(save_path, display_fps=60, save_fps=None, duration=None):
    if save_fps is None:
        save_fps = display_fps
    from glumpy.app.movie import record
    with record(window, save_path, fps=save_fps):
        run(fps=display_fps, duration=duration)
