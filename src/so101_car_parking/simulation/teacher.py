import mujoco
import numpy as np


DT = 0.05


def fingertip(data, fixed=False):
    point = data.geom("fixed_jaw_sph_tip1").xpos.copy()
    return point if fixed else (point + data.geom("moving_jaw_sph_tip1").xpos) / 2


def solve_pose(model, seed, point, rotation, opening, fixed=False):
    pose = mujoco.MjData(model)
    pose.qpos[:] = seed
    pose.qpos[5] = opening
    jp = np.zeros((3, model.nv))
    jr = np.zeros_like(jp)
    for _ in range(140):
        mujoco.mj_forward(model, pose)
        current = fingertip(pose, fixed)
        matrix = pose.body("gripper").xmat.reshape(3, 3)
        angular = 0.5 * sum(np.cross(matrix[:, i], rotation[:, i]) for i in range(3))
        error = np.r_[point - current, 0.08 * angular]
        if np.linalg.norm(error[:3]) < 0.0002 and np.linalg.norm(angular) < 0.003:
            break
        mujoco.mj_jac(model, pose, jp, jr, current, model.body("gripper").id)
        jac = np.vstack((jp[:, :5], 0.08 * jr[:, :5]))
        step = np.linalg.solve(jac.T @ jac + 1e-5 * np.eye(5), jac.T @ error)
        pose.qpos[:5] = np.clip(
            pose.qpos[:5] + np.clip(step, -0.1, 0.1),
            model.jnt_range[:5, 0] + 0.002,
            model.jnt_range[:5, 1] - 0.002,
        )
    mujoco.mj_forward(model, pose)
    alignment = np.trace(rotation.T @ pose.body("gripper").xmat.reshape(3, 3))
    if np.linalg.norm(point - fingertip(pose, fixed)) > 0.002 or alignment < 2.99:
        raise RuntimeError("Unreachable tool pose")
    return pose.qpos[:6].copy()


def hold(data, seconds):
    for _ in range(round(seconds / DT)):
        yield data.ctrl.copy()


def joints(data, target, seconds):
    start = data.ctrl.copy()
    for frame in range(round(seconds / DT)):
        t = (frame + 1) / round(seconds / DT)
        t = t * t * (3 - 2 * t)
        yield start + t * (target - start)


def move(model, data, target, rotation, opening, seconds, fixed=False):
    start = fingertip(data, fixed)
    matrix = data.body("gripper").xmat.reshape(3, 3)
    initial = np.zeros(4)
    final = np.zeros(4)
    mujoco.mju_mat2Quat(initial, matrix.ravel())
    mujoco.mju_mat2Quat(final, rotation.ravel())
    angular = np.zeros(3)
    mujoco.mju_subQuat(angular, final, initial)
    for frame in range(round(seconds / DT)):
        t = (frame + 1) / round(seconds / DT)
        t = t * t * (3 - 2 * t)
        quaternion = initial.copy()
        mujoco.mju_quatIntegrate(quaternion, angular, t)
        orientation = np.zeros(9)
        mujoco.mju_quat2Mat(orientation, quaternion)
        orientation = orientation.reshape(3, 3)
        yield solve_pose(
            model, data.qpos, start + t * (target - start), orientation, opening, fixed
        )


def grasp_contacts(model, data, color):
    sides = set()
    car = model.body(f"{color}_car").id
    for contact in data.contact:
        roots = [
            model.body_rootid[model.geom_bodyid[g]]
            for g in (contact.geom1, contact.geom2)
        ]
        if car not in roots:
            continue
        other = contact.geom2 if roots[0] == car else contact.geom1
        sides.add(model.body(model.geom_bodyid[other]).name)
    return {"gripper", "moving_jaw_so101_v1"}.issubset(sides)


def pick_and_place(model, data, task):
    yield from hold(data, 1)
    color = task["target_car"]
    car = data.body(f"{color}_car")
    position = car.xpos.copy()
    car_rotation = car.xmat.reshape(3, 3).copy()
    cabin = position + car_rotation @ np.array([-0.006, 0, 0.0155])
    bay = next(bay for bay in task["bays"] if bay["color"] == task["target_bay"])
    candidates = []
    ready = data.qpos.copy()
    ready[:6] = [0, -0.7, 1.4, -0.7, 0, 0.55]
    for sign in (1, -1):
        axis = sign * car_rotation[:, 1]
        rotation = np.column_stack((axis, np.cross([0, 0, 1], axis), [0, 0, 1]))
        contact = cabin - 0.0185 * axis
        hover = contact + [0, 0, 0.045]
        try:
            safe = np.array([0.24, 0, 0.14])
            target = solve_pose(model, ready, safe, rotation, 0.55, fixed=True)
            hover_pose = solve_pose(
                model, np.r_[target, data.qpos[6:]], hover, rotation, 0.55, fixed=True
            )
            solve_pose(
                model,
                np.r_[hover_pose, data.qpos[6:]],
                contact,
                rotation,
                0.55,
                fixed=True,
            )
            placement = contact + np.r_[np.array(bay["center"]) - position[:2], 0]
            placement_pose = solve_pose(
                model, ready, placement, rotation, 0.55, fixed=True
            )
            solve_pose(
                model,
                np.r_[placement_pose, data.qpos[6:]],
                placement + [0, 0, 0.055],
                rotation,
                0.55,
                fixed=True,
            )
            candidates.append(
                (np.linalg.norm(target - data.ctrl), target, contact, hover, rotation)
            )
        except RuntimeError:
            pass
    if not candidates:
        raise RuntimeError("No reachable grasp")
    _, target, contact, hover, rotation = min(candidates, key=lambda item: item[0])
    yield from joints(data, target, 2)
    yield from move(model, data, hover, rotation, 0.55, 2, fixed=True)
    yield from move(model, data, contact, rotation, 0.55, 2, fixed=True)
    closed = data.ctrl.copy()
    closed[5] = -0.10
    yield from joints(data, closed, 1.2)
    yield from hold(data, 0.3)
    if not grasp_contacts(model, data, color):
        raise RuntimeError("No bilateral car contact")
    yield from move(
        model, data, contact + [0, 0, 0.008], rotation, -0.10, 0.8, fixed=True
    )
    if car.xpos[2] - position[2] < 0.004:
        raise RuntimeError("Car did not lift")
    grasp_rotation = data.body("gripper").xmat.reshape(3, 3).copy()
    offset = grasp_rotation.T @ (car.xpos.copy() - fingertip(data, fixed=True))
    high = fingertip(data, fixed=True) + [0, 0, 0.047]
    yield from move(model, data, high, rotation, -0.10, 1, fixed=True)
    destination = np.array([*bay["center"], position[2] + 0.004]) - rotation @ offset
    center = np.array([0.26, 0, high[2]])
    yield from move(model, data, center, rotation, -0.10, 2, fixed=True)
    raised = destination + [0, 0, 0.055]
    clearance_x = max(center[0], raised[0], 0.26)
    for waypoint in (
        [clearance_x, center[1], center[2]],
        [clearance_x, raised[1], center[2]],
        raised,
    ):
        yield from move(model, data, np.array(waypoint), rotation, -0.10, 2, fixed=True)
    yield from move(model, data, destination, rotation, -0.10, 2, fixed=True)
    desired = np.array([*bay["center"], position[2] + 0.004])
    for _ in range(2):
        error = desired - car.xpos
        if np.linalg.norm(error[:2]) < 0.0015 or not grasp_contacts(model, data, color):
            break
        correction = fingertip(data, fixed=True) + np.clip(error, -0.012, 0.012)
        yield from move(model, data, correction, rotation, -0.10, 0.7, fixed=True)
    opened = data.ctrl.copy()
    opened[5] = 0.55
    yield from joints(data, opened, 1)
    yield from move(model, data, raised, rotation, 0.55, 1.5, fixed=True)
    yield from hold(data, 1)


def push_routes(model, data, task):
    color = task["target_car"]
    start = data.body(f"{color}_car").xpos[:2].copy()
    bay = next(bay for bay in task["bays"] if bay["color"] == task["target_bay"])
    goal = np.array(bay["center"])
    obstacles = [(np.array(task["robot_mount"][:2]), np.array([0.035, 0.025]))]
    obstacles += [
        (data.body(f"{car['color']}_car").xpos[:2].copy(), np.array([0.044, 0.027]))
        for car in task["cars"]
        if car["color"] != color
    ]
    routes = [
        [start, np.array([goal[0], start[1]]), goal],
        [start, np.array([start[0], goal[1]]), goal],
    ]
    for y in (-0.145, 0.145, -0.19, 0.19, -0.215, 0.215):
        routes.append([start, np.array([start[0], y]), np.array([goal[0], y]), goal])
    right = max(car["position"][0] for car in task["cars"]) + 0.099
    for x in (0.12, 0.22, right, 0.40):
        routes.append([start, np.array([x, start[1]]), np.array([x, goal[1]]), goal])
        for y in (-0.195, 0.195):
            routes.append(
                [
                    start,
                    np.array([x, start[1]]),
                    np.array([x, y]),
                    np.array([goal[0], y]),
                    goal,
                ]
            )
    valid = []
    for route in routes:
        clear = True
        for a, b in zip(route, route[1:]):
            low, high = np.minimum(a, b), np.maximum(a, b)
            for center, half in obstacles:
                extent = half + [0.044 + 0.008, 0.027 + 0.008]
                if np.all(high >= center - extent) and np.all(low <= center + extent):
                    clear = False
        if clear:
            valid.append(route)
    return sorted(
        valid,
        key=lambda route: sum(np.linalg.norm(b - a) for a, b in zip(route, route[1:])),
    )


def robot_contact(model, data, color):
    car = model.body(f"{color}_car").id
    robot = model.body_rootid[model.body("base").id]
    for contact in data.contact:
        roots = {
            model.body_rootid[model.geom_bodyid[g]]
            for g in (contact.geom1, contact.geom2)
        }
        if roots == {car, robot}:
            return True
    return False


def push_to_bay(model, data, task):
    yield from hold(data, 1)
    routes = push_routes(model, data, task)
    if not routes:
        raise RuntimeError("No clear rectilinear push route")
    color = task["target_car"]
    car = data.body(f"{color}_car")
    ready = data.qpos.copy()
    ready[:6] = [0, -0.7, 1.4, -0.7, 0, 0]
    for route in routes:
        try:
            for a, b in zip(route, route[1:]):
                if np.linalg.norm(b - a) >= 0.003:
                    prepare_push(model, ready, np.r_[a, car.xpos[2]], b)
            break
        except RuntimeError:
            continue
    else:
        raise RuntimeError("No reachable push route")
    for nominal, goal in zip(route, route[1:]):
        if np.linalg.norm(goal - nominal) < 0.003:
            continue
        axis = int(np.argmax(np.abs(goal - nominal)))
        fixed = axis == 1
        opening = 1.0 if fixed else 0
        direction = np.zeros(2)
        direction[axis] = np.sign(goal[axis] - nominal[axis])
        initial = car.xpos.copy()
        target, rotation, hover, point = prepare_push(model, ready, initial, goal, axis)
        yield from joints(data, target, 2)
        yield from move(model, data, hover, rotation, opening, 2, fixed)
        yield from move(model, data, point, rotation, opening, 1.5, fixed)
        search_start = car.xpos[:2].copy()
        for _ in range(100):
            point[:2] += direction * 0.00035
            yield solve_pose(model, data.qpos, point, rotation, opening, fixed)
            if (
                robot_contact(model, data, color)
                or direction[axis] * (car.xpos[axis] - search_start[axis]) > 0.0004
            ):
                break
        else:
            raise RuntimeError("No pushing contact")
        offset = car.xpos[:2].copy() - fingertip(data, fixed)[:2]
        point = fingertip(data, fixed)
        previous = car.xpos[axis]
        stalled = 0
        for _ in range(1500):
            remaining = direction[axis] * (goal[axis] - car.xpos[axis])
            if remaining < 0.0015:
                break
            planned = direction[axis] * (point[axis] + offset[axis] - car.xpos[axis])
            if planned < (0.012 if axis == 1 else 0.004):
                point[:2] += direction * min(0.0004, remaining)
            matrix = car.xmat.reshape(3, 3)
            yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
            tangent = np.array([-direction[1], direction[0]])
            centered = goal - offset + tangent * np.clip(0.10 * yaw, -0.012, 0.012)
            other = 1 - axis
            point[other] += np.clip(centered[other] - point[other], -0.0005, 0.0005)
            yield solve_pose(model, data.qpos, point, rotation, opening, fixed)
            progress = direction[axis] * (car.xpos[axis] - previous)
            stalled = stalled + 1 if progress < 0.00002 else 0
            previous = car.xpos[axis]
            if stalled > 80:
                raise RuntimeError("Push stalled")
            if car.xpos[2] > initial[2] + 0.008:
                raise RuntimeError("Car lifted during push")
            if car.xmat.reshape(3, 3)[2, 2] < 0.98:
                raise RuntimeError("Car tipped during push")
        else:
            raise RuntimeError("Push timeout")
        yield from hold(data, 0.3)
        retreat = fingertip(data, fixed) - np.r_[direction * 0.006, 0]
        yield from move(model, data, retreat, rotation, opening, 1, fixed)
        yield from move(
            model, data, retreat + [0, 0, 0.055], rotation, opening, 1, fixed
        )
    yield from hold(data, 1)


def prepare_push(model, seed, initial, goal, axis=None):
    if axis is None:
        axis = int(np.argmax(np.abs(goal - initial[:2])))
    fixed = axis == 1
    opening = 1.0 if fixed else 0
    direction = np.zeros(2)
    direction[axis] = np.sign(goal[axis] - initial[axis])
    extent = 0.042 if axis == 0 else 0.019
    point = np.r_[initial[:2] - direction * (extent + 0.015), 0.019]
    signs = (int(direction[1]),) if axis == 1 else (1, -1)
    upright = [np.array([[0, -s, 0], [s, 0, 0], [0, 0, 1]]) for s in signs]
    rotations = upright.copy()
    for angle in (-0.12, 0.12):
        c, s = np.cos(angle), np.sin(angle)
        lean = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        rotations += [lean @ rotation for rotation in upright]
    options = []
    for rotation in rotations:
        try:
            target = solve_pose(
                model, seed, np.array([0.26, 0, 0.14]), rotation, opening, fixed
            )
            full = np.r_[target, seed[6:]]
            hover = point + [0, 0, 0.055]
            solve_pose(model, full, hover, rotation, opening, fixed)
            solve_pose(model, full, point, rotation, opening, fixed)
            finish = np.r_[goal - direction * extent, point[2]]
            solve_pose(model, full, finish, rotation, opening, fixed)
            cost = np.linalg.norm(target - seed[:6]) + (
                5 if rotation[2, 2] < 0.999 else 0
            )
            options.append((cost, target, rotation, hover))
        except RuntimeError:
            pass
    if not options:
        raise RuntimeError("Unreachable push approach")
    _, target, rotation, hover = min(options, key=lambda item: item[0])
    return target, rotation, hover, point
