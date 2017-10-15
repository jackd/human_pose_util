import scene
from skeleton_vis import limb_collection_animator


time = 0.0

animators = []


@scene.window.event
def on_draw(dt):
    global time
    time += dt
    for animator in animators:
        animator.update(time)


def add_limb_collection_animator(skeleton, poses, fps, linewidth=2.0):
    collection = scene.add_limb_collection(
        skeleton, poses[0], linewidth=linewidth)
    animator = limb_collection_animator(
        collection, poses / scene.pose_scale, fps)
    animators.append(animator)
    return animator


def run(fps=60, duration=None):
    scene.run(fps=fps, duration=duration)


def save(save_path, display_fps=60, save_fps=60, duration=None):
    scene.save(
        save_path, display_fps=display_fps, save_fps=save_fps,
        duration=duration)
