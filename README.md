# Sandbox

**A 3D sandbox for planning, validating and running robot motion paths — and for designing camera moves shot by a real arm.**

Sandbox is a robot motion-planning workspace. You build a move in a 3D simulation, check it against real joint limits and speeds, then upload and run it on the physical arm. Everything a move needs — waypoints, saved poses, reusable path blocks, camera recording and target tracking — lives on one timeline and one 3D scene.

---

## Table of contents

- [What Sandbox is for](#what-sandbox-is-for)
- [The main window at a glance](#the-main-window-at-a-glance)
- [Basic controls](#basic-controls)
- [List of Elements](#list-of-elements)
  - [3D objects](#3d-objects)
  - [Waypoints](#waypoints)
  - [Bookmarks and saved points](#bookmarks-and-saved-points)
- [Control panel](#control-panel)
- [Path baking](#path-baking)
- [Simulate and Run](#simulate-and-run)
- [Timeline](#timeline)
  - [Linear speed](#linear-speed)
  - [Transitions](#transitions)
- [Saved paths](#saved-paths)
- [Object properties](#object-properties)
- [Components](#components)
  - [Camera](#camera)
  - [Look At](#look-at)
  - [Offset](#offset)
  - [Collider](#collider)
- [Miscellaneous features](#miscellaneous-features)
- [Feature map](#feature-map)
- [Current limitations](#current-limitations)
- [Project layout](#project-layout)

---

## What Sandbox is for

- **Plan a move in 3D.** Lay down waypoints by hand in the simulator or by capturing poses from the real arm, then shape the path with Bezier curves.
- **Check it before you shoot or run it.** Baking samples the path, verifies every sample point is a reachable pose, and flags speed problems — so a bad move is visible before the arm ever moves.
- **Run it on the real robot.** Upload the baked path, move the arm to its init pose, and run. Simulate and Run are two separate steps by design.
- **Reuse work.** Saved points and whole saved paths can be stored per project or globally, then dropped onto any timeline.
- **Add cameras and targets.** Cameras can be simulated, USB/OpenCV or Sony, each with its own record track. A Look At component keeps the arm aimed at a designated object along the path.

---

## The main window at a glance

| Panel | What it does |
| --- | --- |
| **Main View** | The 3D world. Robot, objects, waypoints and paths, plus camera previews. |
| **List of Elements** | Every waypoint, bookmarked pose and 3D object in the scene, in tabs, with per-item and global visibility toggles. |
| **Control Panel** | Bake, simulate, run, and hand the real arm over to the simulator's pose. |
| **Timeline** | The move over time: an arm track and a camera track, with waypoint frames and generated sample points. |
| **Speed Graph** | How fast the arm travels along the path. |
| **Object Properties** | Transform, joint poses and components for whatever is selected in the world view. |

---

## Basic controls

- **Left-click drag** in the 3D view to orbit the camera around the scene.
- **Right-click drag** to pan.
- **Select the EE controller** — the simulator's robot cursor — to bring up a transform gizmo.
- **Drag the gizmo axes** to move and rotate the robot pose. Inverse kinematics is solved in real time; if IK fails, the robot freezes until a solution is found again.
- **Switch axes mode** between *world axes* (absolute positional changes) and *local axes* (changes relative to the current end-effector pose).

---

## List of Elements

The elements list shows everything present in the scene, grouped into tabs:

- **Waypoints** — points that build the robot's motion path
- **Bookmarked positions** — saved important poses
- **3D objects** — imported and primitive geometry

Add items with the **+** button in the relevant tab. Each item has its own visibility toggle (the **eye** button), and there is a global "toggle all" control. Item counts are shown per tab.

### 3D objects

- The simulated robot appears as the **sim cursor** — the draggable handle that stands in for the real arm's end effector.
- Add simple primitives, or import custom **`.stl`** meshes to build a simulation environment around the robot.
- Add advanced objects such as **cameras**, to reproduce a real camera's position in the simulated world.
- **Parent hierarchy.** Objects can be parented to one another so a child moves with its parent. This is how you bind objects to the end effector — an EE-parented camera, for example, follows every move the arm makes.

### Waypoints

Waypoints are the points that define the robot's motion path in 3D space. They can be added from two sources:

- **From the real robot pose (orange)** — capture where the arm physically is. Good for finding a shot on set.
- **From the sim bot pose (blue)** — capture where the simulated arm is. Good for defining positions inside the simulator.

Both control styles switch freely, so you can set some points on the real arm and others in the simulator within a single path.

**Why curves instead of hand-dragging.** Recording a genuine hand-drag would mean capturing a point every few milliseconds — well over 200 points for a simple path — which also records every hand tremor and every wobble in speed and timing, and leaves 200 points that each need fine-tuning. Sandbox solves this with **Bezier curve path design** instead:

- Click a waypoint to reveal its curve handles.
- Drag the handles to shape the path between waypoints.
- The result is a clean, smooth curve that stays easy to adjust — nudging one waypoint does not distort the whole move.

### Bookmarks and saved points

**Saved points** store important poses for the sim bot to return to.

- Add them with the same logic as waypoints (real-arm pose or simulator pose).
- Click the **arrow** button to teleport the sim bot straight to that pose.
- Two poses are provided by default: **Home pose** (the right-angle pose) and **Real pose** (where the real arm currently is).

In the 3D view, **saved points are pink** and **waypoints are white**; the arm's forward direction and right-hand direction are shown for orientation.

Bookmarks can be stored **locally per project** or **globally** so they are shared across multiple files.

---

## Control panel

| Control | Function |
| --- | --- |
| **Move real robot to sim bot pose** | Sends the real arm to wherever the simulator's cursor is. Linear speed is adjustable in Settings. |
| **Bake path** | Generates the sample points that approximate the path and validates them (see below). |
| **Simulate** | Plays the baked path in the 3D world. |
| **Run** | Runs the path on the real robot. |
| **Repeatable** | Marks the motion path as repeatable *(untested)*. |

---

## Path baking

Baking converts a hand-shaped path of waypoints into a dense set of evenly spaced sample points the robot can actually follow, and checks that it is safe. It does three things:

1. **Generates sample points** to approximate the path. Points are spaced by **distance**, not time — segment timing is adjusted afterwards on the timeline, so baking does not have to be redone just to change speed.
2. **Checks reachability (IK).** Every sample point must be a valid pose for the arm.
3. **Checks speed.** Sample points must be spaced and timed so the arm never exceeds its limits.

> **Any change to the timeline, or to a waypoint's position, requires a full rebake.**

### Bake states

Baked points are colour-coded in the 3D view:

| Colour | Meaning |
| --- | --- |
| White / grey | Not yet baked |
| Red | IK failed — pose not reachable |
| Green | Everything OK |
| Blue | Force FK mode (see below) |
| Orange | IK is fine, but the arm would be moving too fast |
| Yellow | IK checked, speed not yet checked |

### Force FK mode

Some moves care more about the **destination** than the shape of the path to it — and some waypoint arrangements are simply too complex for IK to solve reliably in between. Marking a waypoint segment **Force FK** skips sample-point generation inside that segment entirely and issues a direct **moveJ** between the two waypoints.

---

## Simulate and Run

Both actions only become available after a clean bake.

**Simulate**
- Plays the motion path in the 3D world view.
- **Pause** is available.
- Drag the timeline cursor to replay any part of the path.

**Run**
- **Uploads the path to the robot controller first** — the arm does not move until the upload is complete.
- Press and hold the button to move the robot to its init pose.
- Then run the path.
- **Pause is not available** during a run, but the run can be interrupted.

---

## Timeline

The timeline is where a move is timed and refined.

**Arm track**
- **Diamond frames** are your waypoints.
- **Circle frames** are the sample points generated by baking.
- On the first bake, sample points are spaced evenly. Drag them to customise speed, then rebake to run the speed check again.

**Camera track**
- **Green diamond** — start recording.
- **Red diamond** — stop recording.
- Note that there is a delay in sending the command to a Sony camera, so the recorded range can differ slightly from the marked one.

A **snap-to-grid toggle** is available for frame placement — snap to the grid for simplicity, or place frames freely for fine tuning. A **cursor** marks the current position in time.

### Linear speed

For irregular paths that still need a constant travel speed, select a range of waypoints and apply the **linear speed** option. Sandbox measures the distance of the selected path segments and approximates the time needed for the arm to cover them at the speed you specify, setting those segments to run at a constant speed.

### Transitions

Rather than dragging every sample point by hand, you can apply a transition to control a segment's acceleration:

- **Custom drag** — shape it yourself on the timeline
- **Linear** — constant speed
- **Ease in** — slow to fast
- **Ease out** — fast to slow

Select a start waypoint, the sample points and an end waypoint, and adjust the curve between them. Transitions can also be customised with **Bezier curves** *(untested)*.

> Transitions are spaced mathematically and do not yet take path distance or the robot's maximum speed into account, so a short, fast segment can end up very fast. A safety check for this is in progress.

---

## Saved paths

A **saved path** is an entire timeline segment captured as a single reusable block.

- Save locally (per project) or globally (shared across files).
- Drag a saved path onto the timeline to reuse it, or load one of the provided default paths — handy for users who just want a ready-made move.
- Saved paths and regular waypoints can be used together in the same timeline.
- Segments connected to a saved path default to **Force FK** mode, which keeps things simple for users who rely on ready-made paths.
- Experienced users can **unwrap** a saved path back into individual waypoints for detailed edits.

Hovering over a saved path reveals the original waypoints it was built from.

---

## Object properties

The properties panel shows the transform and other relevant data for whatever is selected in the world view.

**Cursors (the robot)**
- Name, EE position and rotation, and joint poses.
- Transforms are editable, and drive the simulated robot — they do not move the real arm.

**3D objects**
- Name, position and rotation (in world space, or relative to a parent if one exists), and scale.
- A reset button (↻) at the end of the position row and of the rotation row sets those values back to 0.
- When the parent has an enabled Offset component, **snap object to parent offset** appears under the rotation row. It moves the object onto the offset once and switches it to Local; it does not keep following.

**Waypoints and saved points**
- Name, the EE position and rotation needed to reach the point, and joint poses.
- Transforms can be edited to move waypoints, but not saved points.

**Angle and orientation handling** — most angle properties convert between **radians and degrees**, and orientation properties convert between **Euler angles and quaternions**.

---

## Components

Components attach extra behaviour to a 3D object — turning plain geometry into something functional.

### Camera

Adding a **Camera component** converts a 3D object into a working camera and automatically generates its view, its track on the timeline, and its properties panel.

Three camera types are supported:

| Type | Use |
| --- | --- |
| **Sim** | Preview the camera inside the simulator and rehearse the video shoot against simulated 3D objects. |
| **OpenCV** | Attach a custom Python script for USB webcams and advanced tracking — useful for vision work beyond shooting. |
| **Sony** | Control a real Sony camera through the Sony SDK. |

Camera recording controls:
- **Record keyframes** on the timeline to define when the camera rolls.
- A **manual override toggle** records or stops independently of the timeline keyframes.
- Explicit **start record** and **stop record** buttons.

#### Script targets

An OpenCV camera script can publish targets as module variables, read every frame:

- `primary_target = [x, y, z]` — shown as a large red sphere.
- `targets = [[x, y, z], ...]` — shown as small yellow spheres.

By default each point is `[x_px, y_px, depth_m]` in the camera image, the same format as the older `tracked_cam_pos` (which is still read as `targets` when a script has no `targets`). Set `target_space = "camera"` for metres in the OpenCV camera frame (x right, y down, z forward), or `"world"` for world metres. All points are listed in the **Tracked Points** tab of the hierarchy, primary target first; its eye button hides the spheres.

### Look At

*(Formerly "Tracker" / "Track Object".)* Given a target, the sim cursor keeps **looking at** a designated object — both while you drag it by hand and along the path segments it generates. Combined with an OpenCV camera, the arm can track and centre real targets as well as simulated ones *(in progress)*.

Move the target object in the simulator and the arm's orientation follows it, so aiming does not depend on whether face detection succeeded on any given run.

> Tracking is computed from the target's position **before** baking. The arm does not follow a moving target during a run yet — live tracking is in progress.

The aiming point sits at the **Pos offset** (cursor-local) and the cursor is turned by the **Rot offset** after aiming. **copy offset component values** fills both fields once from the cursor's Offset component, so that the Offset frame is what aims at the target (for example a lens or tool tip mounted off-centre). It is a one-time copy: later Offset edits don't change Look At. A red warning appears, and the button is disabled, while the Offset component is missing or switched off. Waypoints added from the cursor keep the Look At values they were added with.

**🎯 select primary target** (right of the pick button) makes Look At follow the OpenCV camera script's primary target (the big red sphere) instead of an object. The sim cursor re-aims live as the target moves; if the script loses the target for a frame, it keeps aiming at the last place it was seen.

Adding Look At also adds a **Look At** track to the timeline. On each bake it shows a bar, labelled with the target, over every stretch of the path that is aimed at a target (a span is aimed when either of its waypoints has Look At on). Stretches with no target stay empty. The bars only change when you bake.

### Offset

The **Offset component** can be added to any 3D object and to the sim cursor. It adds a yellow, waypoint-like frame fixed to its owner, shown only while the owner is selected. Click the yellow sphere, or click the Offset panel in Properties, to get its move/rotate gizmo (click the owner again to get the owner's gizmo back; deselecting hides it). It never joins the waypoint list or the path.

The properties panel shows its **local** position and rotation relative to the owner's origin; edit them there or with the gizmo. New offsets start 10 cm along the owner's local z axis. Offsets are saved with the project and covered by undo for scene objects.

### Collider

The **Collider component** can be added to any 3D object and to the sim cursor.

- **Sim cursor:** the collider covers the whole sim robot in the shape of its URDF (one convex hull per link). Its shape and size can't be changed; **Margin** (default 0 m) makes it thicker.
- **Other objects:** the component holds a list of collider boxes. Each box has a shape (**Cube** or **Sphere**), a position and rotation relative to the object, and a size in meters (a diameter for spheres). The size does not follow the object's scale. A new box starts wrapped around the object as it is at that moment. Use **+** to add a box and the **−** at the top right of a box to delete it; several boxes on one object make a custom shape. Rotation has the same Quat/Euler and rad/deg toggles as Offset.
- **Green outlines** of the boxes (or of the robot hull) show only while you click the Collider panel. Selecting the object again hides them. Collider boxes can't be moved in the 3D view yet.
- **During a simulation**, when colliders of two different objects touch, both models turn red. Boxes of the same object never count against each other. A parent and its child do collide (the arm can hit its own end effector) unless the child ticks **ignore parent collider**, which shows for parented objects and skips the nearest parent above it that has a collider (for a tool on the sim cursor, the robot). If either object has **stop simulation if collide** ticked, the simulation pauses (same as the Pause button) when the contact starts; Resume plays on while they stay touching.

Colliders are saved with the project and covered by undo for scene objects.

Components in the properties panel are listed in the order they were added, newest at the bottom.

---

## Miscellaneous features

- **File save and load**, with **autosave and automatic backup**.
- **Settings menu** for robot type and joint limits.
- **Automatic reconnection** to the robot.
- **Multi-camera handling** — the simulator, computer webcams and Sony cameras are handled separately, with split-screen views and an individual recording track for each.
- **View console** for debugging.
- **Path window** — render only a range of waypoints in the 3D viewer at a time, to keep busy scenes readable.
- **View cube** — displays and controls the camera's viewing angle, as in CAD software.
- **Dev mode toggle** with a password lock, to hide development features from regular users.

---

## Feature map

| Function | What Sandbox provides |
| --- | --- |
| **Timeline keyframes** | Waypoints can be re-timed and reordered; moving one point's timing does not scramble the order of the rest of the path. |
| **Path shape and 3D preview** | Bezier curves drawn directly in the 3D view, with a simulated camera view — so you can judge framing before shooting instead of reasoning about joint angles. |
| **Finding points on set** | Two interchangeable control styles: drive the real arm to a position and capture it, or place the pose in the simulator. Switch between them freely. |
| **Simple path building** | Save and open your own paths, or load factory example paths to start from something that works. Reuse one move across scenes. |
| **Teaching by hand-dragging** | Only confirmed points are captured, joined by curves — so hand tremor, uneven speed and drag timing never end up in the path. Adjusting one point does not drag the whole gesture out of shape. |
| **Speed ramps** | Each segment can be linear or use a custom transition curve — slow the start or stop hard on one segment without redrawing its neighbours. |
| **Target lock** | Path points can be bound to a target in the scene so the arm's orientation follows it, independent of whether per-run recognition succeeds. |
| **Vision-based following** | The OpenCV camera can compute a 3D point from selected pixels with a depth camera for 3D tracking, and can run custom external scripts — not just face detection, but tasks like parcel recognition. |
| **Bake debugger** | Every segment is checked for joint and linear velocity limit violations, and speed logs can be exported — so an overspeed is visible before the shoot rather than as a shaky take. Simulation and real-run data can be exported to diagnose speed and position problems. |
| **Separate upload and run** | The path transfers to the controller first and only then runs; the stop control stays inactive until the transfer finishes — so the software clock never runs ahead of the physical arm. |
| **Scene hierarchy** | Objects can be parented, with local or global coordinates, and the sim cursor can act as a parent — letting you mount additional items and execute a wider range of tasks. |
| **Multi-camera** | Simulated, computer webcam and Sony cameras as separate types, with split-screen views and per-camera recording tracks: which camera is recording, and from which second to which second, is written on the timeline. |
| **Virtual production output** | A simulated camera inside the software. |

---

## Current limitations

The following are known to be rough or incomplete at this stage:

| Area | Status |
| --- | --- |
| Speed check during baking | Currently passes regardless of the path. |
| Bake "yellow" state (IK checked, speed not yet checked) | Unreliable. |
| Transitions | Spaced mathematically without regard to path distance or the arm's maximum speed; the safety check is in progress. |
| Bezier-customised transitions | Untested. |
| Repeatable motion path | Untested. |
| Timeline snap-to-grid | Not working. |
| Path window (rendering a range of waypoints) | Not working. |
| Real-robot cursor in Object Properties | Displays incorrectly. |
| Live target tracking during a run | Not available; tracking is computed before baking. Real-arm face tracking is in progress. |
| Sony camera control | Noticeably laggy when driven through this program; for real shoots, use the dedicated Sony workflow. |
| Automatic reconnection | Laggy. |

---

## Project layout

Run the app with `python backend/webui.py`. Every file the app reads or writes is resolved in `backend/app_paths.py`, so it works from any working directory. The tree is split three ways with a future Electron build in mind:

```
backend/                  everything needed to run the app
  webui.py                Flask web UI (entry point)
  app_paths.py            folder constants
  templates/index.html    the page
  cameras/                sony_liveview.py, libusb-1.0.dll (Sony camera support)
  state/temp.json         autosave (internal, not for hand editing)
  robot_sim/
    robot_sim_run.py      Viser viewer loop and request queues
    robot_runtime.py      sim/real robot state, IK, Simulate / Run, path buffer
    waypoints.py          waypoints, Bezier handles, bookmarked points, saved paths, selection, undo/redo
    simulatorview.py      other scene objects: primitives, meshes, parenting, measure, cameras
    path_bake.py          path sampling, IK and speed checks, baking
    timeline.py           timeline control: keyframe drag/snap, sample times, transitions, copy/paste
    file_manage.py        save / open / export
    arcscontrol.py        AUBO SDK wrapper
    aubo-i5-full-master/  robot URDFs and meshes
user_files/               ships with the app, editable by the user
  settings.json           settings, global saved points and paths
  cam_scripts/            OpenCV / RealSense scripts added from the GUI
  paths/                  saved .arcspath projects
  exports/                export.json, bake_overspeed.csv / .xlsx
  recordings/             camera recordings
  imported_meshes/        meshes added with Import mesh
docs/                     not needed at runtime
  examples/cam_scripts/   example camera scripts (YOLO / RealSense), models/ holds yolov8n.pt
  CameraRemoteCommand-2.00.02/   Sony PTP SDK reference and examples
  archive/, demo/, export-format/, Function List, slides, SDK manual, bug log
```

A packaged build can move the writable folders with the `ARCS_USER_DIR` (for `user_files`) and `ARCS_STATE_DIR` (for `backend/state`) environment variables.

Functions moved into `simulatorview.py` and `timeline.py` are still reachable through their old module (for example `wp.add_scene_object` or `bake.pin_key_times`), so existing calls keep working.
