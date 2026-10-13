"""Recorded-state container-lift helper for the DINO evaluation launcher."""


def _counter_top_under(sim, x, y):
    """Return the counter or island top under an xy position."""
    model = sim.model
    best = None
    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] != 6:
            continue
        geom_name = model.geom_id2name(geom_id) or ""
        if "counter" not in geom_name and "island" not in geom_name:
            continue
        geom_x, geom_y, geom_z = sim.data.geom_xpos[geom_id]
        size_x, size_y, size_z = model.geom_size[geom_id]
        if abs(geom_x - x) <= size_x and abs(geom_y - y) <= size_y:
            top = geom_z + size_z
            if top < 2.0 and (best is None or top > best):
                best = top
    return best


def _compute_container_lifts(env):
    """Compute the archived render correction for containers clipping counters."""
    raw_env = env.env
    sim = raw_env.sim
    model = sim.model
    lifts = {}
    for name, obj in raw_env.objects.items():
        if "container" not in name:
            continue
        body_id = raw_env.obj_body_id[obj.name]
        joint_address = model.body_jntadr[body_id]
        if joint_address < 0 or model.jnt_type[joint_address] != 0:
            continue
        qpos_z_address = model.jnt_qposadr[joint_address] + 2
        geom_z = [
            sim.data.geom_xpos[geom_id][2]
            for geom_id in range(model.ngeom)
            if model.geom_bodyid[geom_id] == body_id
        ]
        if not geom_z:
            continue
        x = sim.data.qpos[qpos_z_address - 2]
        y = sim.data.qpos[qpos_z_address - 1]
        top = _counter_top_under(sim, x, y)
        if top is None:
            continue
        lift = top - min(geom_z)
        if lift > 0.002:
            lifts[qpos_z_address] = lift
    return lifts
