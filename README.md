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
- [Gimbal (DJI RS4)](#gimbal-dji-rs4)
- [Miscellaneous features](#miscellaneous-features)
- [Game controller (Xbox / PlayStation)](#game-controller-xbox--playstation)
- [Steam Deck edition](#steam-deck-edition)
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
- **Aubo Arm** is the robot base, shown as a blue marker at the foot of the arm. The **Sim cursor** and, while the robot is connected, the **Real cursor** sit under it in the hierarchy. Select it to get a gizmo at the base, or type its position and rotation in Properties: both the real and the sim arm move with it, and the sim cursor keeps its joint angles. IK values, MoveL targets and the IK fields stay relative to the arm's base, as on the teach pendant; saved points (Bookmarks) always move with the arm. After a move, if there are waypoints, the app asks whether they should **move with the arm** or **stay in place**. Waypoints that stay turn red and lose their joint angles; the next bake solves them again and turns each one white once IK is found. The base pose is saved with the project and can be undone (together with the waypoint choice).
- **More Aubo Arms.** *Add object > Aubo Arm* adds another arm, named **Aubo Arm 1**, **Aubo Arm 2** and so on, placed beside the last one (the first arm keeps the name **Aubo Arm**). Each arm has its own base, Sim and Real cursors, IK (relative to its own base) and waypoints. Objects, cameras and saved paths are shared, in one world frame. Saved points are shared too, but relative to the base: each one is drawn once in front of every arm, and going to it moves the active arm to its own copy. Selecting an arm's cursor or base, clicking its timeline row, clicking one of its waypoints, or picking it in *Preferences > Robot Settings* makes it the **active arm**, named in brackets in the Waypoints header. The Waypoints list shows every arm's waypoints, each arm under its own name, and the **+** buttons add to the active arm. Double-click an arm in the object list to rename it; select an added arm and press Delete to remove it (the first arm stays). Bake, Simulate and Run work on every arm that has a path, on one clock.
- Add simple primitives, or import custom **`.stl`** meshes to build a simulation environment around the robot.
- **Empty** objects are drawn as grey, see-through waypoint markers (sphere with forward and side arrows), so their orientation is visible.
- Add advanced objects such as **cameras**, to reproduce a real camera's position in the simulated world.
- Add a **Gimbal** to drive a DJI RS4 gimbal over CAN from the timeline (see [Gimbal (DJI RS4)](#gimbal-dji-rs4)).
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
- Press and hold the button to move the robot to its init pose. Every Gimbal with frames moves to its initial pose (its track at 0 s) at the same time.
- Then run the path.
- **Pause is not available** during a run, but the run can be interrupted.

---

## Timeline

The timeline is where a move is timed and refined.

**Track names.** Drag the right edge of the track-name column to widen or narrow it (the width is remembered in this browser). Right of the names is a column of small **+** buttons, the same as the lists' **+**: on an arm's row, orange adds a waypoint from the Real cursor and blue from the Sim cursor at the playhead (on another arm's row, that arm becomes the active arm first); on a gimbal's row, orange inserts a frame with the real gimbal's angles and blue with the Sim gimbal's; on a camera's row, the green ▶ and red ❚❚ add a start / stop recording key at the playhead. Only tracks you added (dev mode **+** under the names) have a 🗑; camera, gimbal and Look At tracks go with their object or component.

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

The aiming point sits at the **Pos offset** (cursor-local) and the cursor is turned by the **Rot offset** after aiming. **Offset from tool** lists the objects parented to the Sim cursor (swappable EE tools such as a lens or gripper tip). Pick one from the list, or press the ○ button beside it and click the child in the 3D view or the object list: its **local** transform is copied once into the pos and rot offset, so that child is what aims at the target. Later moves of the child don't change Look At; editing the offset fields by hand drops the tool name. Waypoints added from the cursor keep the Look At target, offset and tool they were added with, so the tool can change along the path.

**🎯 select primary target** (right of the pick button) makes Look At follow the OpenCV camera script's primary target (the big red sphere) instead of an object. The sim cursor re-aims live as the target moves; if the script loses the target for a frame, it keeps aiming at the last place it was seen.

Adding Look At also adds a **Look At** track to the timeline. On each bake it shows a bar over every run of spans aimed at the same target with the same offset, labelled with the target and the tool (or "offset" when the offset was typed in). A change of offset starts a new bar even when the target stays the same. A waypoint with Look At on that shares it with neither neighbour gets a thin bar with no label; hover any bar to see its target, tool and offsets. Stretches with no target stay empty. The bars only change when you bake.

### Offset

> The Offset component is switched off for now: it can't be added, and existing Offset data in files is kept but not shown or used.

The **Offset component** can be added to any 3D object and to the sim cursor. It adds a yellow, waypoint-like frame fixed to its owner, shown only while the owner is selected. Click the yellow sphere, or click the Offset panel in Properties, to get its move/rotate gizmo (click the owner again to get the owner's gizmo back; deselecting hides it). It never joins the waypoint list or the path.

The properties panel shows its **local** position and rotation relative to the owner's origin; edit them there or with the gizmo. New offsets start 10 cm along the owner's local z axis. Offsets are saved with the project and covered by undo for scene objects.

### Collider

The **Collider component** can be added to any 3D object and to the **Aubo Arm** (the sim robot's base). The sim cursor has no Collider of its own: it is part of the Aubo Arm's robot collider. Each Aubo Arm has its own.

- **Aubo Arm:** the collider covers the whole sim robot in the shape of its URDF (one convex hull per link). Its shape and size can't be changed; **Margin** (default 0 m) makes it thicker.
- **Other objects:** the component holds a list of collider boxes. Each box has a shape (**Cube** or **Sphere**), a position and rotation relative to the object, and a size in meters (a diameter for spheres). The size does not follow the object's scale. A new box starts wrapped around the object as it is at that moment. Use **+** to add a box and the **−** at the top right of a box to delete it; several boxes on one object make a custom shape. Rotation has the same Quat/Euler and rad/deg toggles as Offset.
- **Green outlines** of the boxes (or of the robot hull) show only while you click the Collider panel. Selecting the object again hides them. Collider boxes can't be moved in the 3D view yet.
- **During a simulation**, when colliders of two different objects touch, both models turn red. Boxes of the same object never count against each other. A parent and its child do collide (the arm can hit its own end effector) unless the child ticks **ignore parent collider**, which shows for parented objects and skips the nearest parent above it that has a collider (for a tool on the sim cursor or the Aubo Arm, the robot). If either object has **stop simulation if collide** ticked, the simulation pauses (same as the Pause button) when the contact starts; Resume plays on while they stay touching.

Colliders are saved with the project and covered by undo for scene objects.

Components in the properties panel are listed in the order they were added, newest at the bottom.

---

## Gimbal (DJI RS4)

*Add > Gimbal* (or **+** in the Objects tab) adds a gimbal, named **Gimbal**, then **Gimbal 1**, **Gimbal 2** and so on. Like an Aubo Arm with its Sim and Real cursors, it has a mount and two twins, listed under it:

- **Gimbal** (the mount) — the gimbal's **handle** (grey). This is what you move, and it stores the handle's pose: gizmo, Properties transform, parenting (for example to the Sim cursor, so the gimbal rides on the arm). Unrotated, the hand end faces up and the camera end down.
- **Sim gimbal** — the gimbal's arms and motors built from boxes, in blue, with a long lens arrow at the camera, posed by the **target** angles: what you set, and the track when the playhead moves. A new gimbal gets a **Camera** object (*<name> camera*) parented to the Sim gimbal at its camera centre, so it turns with the gimbal: give it the real camera's type (sim, OpenCV, Sony). Other objects can be parented to the Sim gimbal too (drag them onto its row). A new gimbal's Sim gimbal starts at the real gimbal's pose: its target takes the first angles read (unless you set the target or add frames first).
- **Real gimbal** — the same in orange, posed by the angles **read from the gimbal**. The app tries to read them all the time (about 10 times a second, reconnecting every 2 s). The Real gimbal is drawn, and listed, only while reading works; when a read fails it disappears and red text under the Sim gimbal's properties says why.

The twins can't be moved, renamed or deleted on their own; their angles come from the target and from the gimbal.

Angles are the gimbal's three motors, **J1–J3**, in degrees, numbered from the handle: **J1** the handle motor, **J2** the middle motor, **J3** the camera motor (the DJI SDK calls them joint yaw / roll / pitch; saved files keep those keys). Values are kept as read, in ±180 (the SDK's ±30 / −56…+146 are limits for move commands, not for the joints: an RS4 with the camera looking along the handle reads J3 = −90). They are motor angles, fixed to the gimbal, not world directions, so they mean the same however the gimbal is mounted.

**Handle and camera.** The mount stores the handle's pose; J1–J3 store the camera's orientation relative to the handle. With the mount unrotated (hand end up, camera end down) and J1–J3 at 0, the camera looks along the front arrow, level, at right angles to the handle. J3 is tilt. The values shown are the RS 4's own readings, which the model file maps to model angles (`sign`, `offset`): on the RS 4 a rising J3 reading tilts the lens away from the hand end (the model file's J3 `sign` is −1), and the level camera reads about J3 −69 with J1 / J2 near 180. **J1** turns the camera around the handle axis (the 360 roll when J3 = −90), J2 around the lens. To fit the gimbal to the arm, parent it to the Sim cursor and rotate the mount to how the handle is fixed to the flange: the example project *gimbal pose* has the hand end towards the robot and the camera end forward, so a level camera (about J3 −69) looks straight ahead.

**Gimbal models.** The shape, the joint axes and the offsets are not built into the sandbox: they come from a **model file** in `user_files/gimbal_models/`, picked in the Gimbal's **Model** list (saved with the project by file name). `dji_rs4.json` is the DJI RS 4. A file holds, in millimetres in the handle's frame with all joints at 0:

- **joints**, from the handle out: each with an **axis** direction (any direction, so a joint that is not at right angles to the one before, like the RS 4's J2, is just a different vector, e.g. `[1, 0, -0.2]`), a **point** on the axis, which reading it is (`sdk`: yaw / roll / pitch) and its range;
- **aim_joints**: the two joints Look At may turn;
- **camera**: its centre and lens and right directions (no camera body is drawn: the Gimbal's Camera object sits there);
- **parts**: boxes, bars and spheres to draw, each on a **link** (0 = the handle, drawn grey; 1, 2, 3 = moving with that joint);
- **driver**: the default driver script.

On a balanced gimbal all three axes pass through the camera's centre of gravity, so the camera's direction depends only on the axis directions and its position only on the camera centre's place below the pan motor; the arms only change the look. DJI publishes no CAD or axis dimensions for the RS 4 (only its overall size, about 370 × 191 × 189 mm unfolded), so the sizes in `dji_rs4.json` are estimates: measure your gimbal (camera centre below the pan motor, arm positions) and edit the file, or copy it for another gimbal or rig. J2's axis on the RS 4 is not square to J1: the file has it tilted 56° from the lens line towards the hand end (34° from the handle axis), fitted to readings with the camera level and J1 / J2 near 180°; a gimbal whose Real twin looks up or down while the real camera is level needs this tilt checked.

**Joints / World.** The switch beside the Sim gimbal's **Target** (and the Real gimbal's readout) shows the angles as the motors' **J1–J3** or as the camera's **world** yaw (heading), pitch (lens above the horizon) and roll (picture turned, positive = right side down). Typing world angles solves J1–J3 for them (inverse kinematics, from the current angles so the gimbal doesn't flip); frames and the track are still joint angles. If the gimbal can't reach that orientation it goes to the nearest one and says how far off it is.

**Turning the camera turns the gimbal.** Rotating the Sim gimbal's Camera with its gizmo, or in Properties with **Global** on, solves J1–J3 so the gimbal points the camera that way, and the model moves with it (the camera stays on the gimbal, at the nearest orientation it reaches). Rotating it with **Local** on sets the camera's offset on the gimbal instead. Not while Look At aims the gimbal.

Joints to world needs the handle's pose (the mount: on the arm, the flange) and the model's axes; world to joints also needs them to be right, J2's tilt above all: with the tilt in `dji_rs4.json` the model reaches only part of the orientations a real RS 4 does, which suggests that tilt is wrong; Recalibrate fits it properly from many readings.

**Recalibrate gimbal.** Fits the model to your gimbal from many readings. With it connected (the Real gimbal showing):

1. Hold the gimbal with the **handle vertical, hand end up**, and keep the handle still (a stand or clamp helps; which way it faces around the handle doesn't matter).
2. Press **Recalibrate gimbal**. The button becomes **Finish recalibration (N samples)** and a **Cancel** button appears.
3. Move the camera around with the gimbal's joystick for 10 s or more: pan all the way round (J1) and tilt up and down (J3). Every tick (10 Hz) in which the joints moved, the sandbox records J1–J3 and the gimbal's own world yaw, roll and pitch (attitude). The note under the button shows the sample count and the J1 / J3 spans; aim for J1 over 180° and J3 over 90°.
4. Press **Finish**. The sandbox fits, together: J2's axis (its tilt towards the hand end and its sideways lean), J2's and J3's direction and zero offset, and how far off vertical the handle was held. This takes up to a minute (Fitting…).

The result goes into the model file (`sign`, `offset` and `axis` on J2 / J3, with a `calibrated` note) and the gimbal redraws at once. Before writing, the file is copied to `<file>.<date-time>.bak`, and the first time also to `<file>.orig`, which is never overwritten; every fit starts from the joints in `.orig`, so a bad result doesn't steer the next one (to start over, copy `.orig` back, or restore the file from git). Each recording (all readings and the fit) is saved in `user_files/gimbal_models/calibrations/`, for checking by hand. A fit with an rms of 4° or more is not saved. The note under the button gives the fit quality (rms and worst error over all samples, in degrees: under 1.5 is good; a poor fit usually means the handle moved), the new and previous values, and can be selected and copied. Different sign / offset combinations can describe the same gimbal; of those that fit equally well, the one nearest the model is kept. The readings can't tell the gimbal from its mirror image, which pans the same way but tilts up for down, so **J3's direction is not fitted**: it is the J3 `sign` in the model file (in `dji_rs4.json` it is -1: the RS 4's J3 reading runs against the model's J3 axis). If the Sim gimbal tilts the opposite way to the real one, flip J3's `sign` in the model file (and its `offset` to keep the level reading level, or just recalibrate) — the fit always keeps the file's J3 sign. A model file that has never been recalibrated (new, restored or edited) replaces `.orig` at the next recalibration. The signs of the world pitch and roll readings are fitted instead (the note gives the best rms for each). The heading's zero is free (it drifts), so J1's zero stays as it is. The box shapes in the model file are not changed, so after a large change of J2's tilt the drawn arms may not line up with the axis. The driver needs `get_world()` (the bundled `rs4_can.py` has it). The sandbox re-reads a model file when it changes, so edits show at once; a file it can't read shows its error in red under **Model** and the gimbal is drawn as plain axes.

**Properties**

| Selected | Shows |
| --- | --- |
| **Gimbal** | Name, transform, scale, components; **Model** (the model file); **Recalibrate gimbal** (fit J2's axis and the J2 / J3 directions and offsets from recorded readings, see above); **Off / Drive** (send moves to the real gimbal: Move gimbal, the initial pose and Run; angles are read either way); the connection state; **Connection**: driver script (empty = the bundled `rs4_can.py`), python-can interface, channel, bitrate and *Move time* (for Move gimbal and the move to the initial pose). |
| **Sim gimbal** | **Collider** (+ menu: shaped by the model's arms and motors, not the handle, so the gimbal on the flange doesn't count as hitting the arm; margin, stop simulation if collide); **Target** J1 / J2 / J3 (the moves you type; the track sets them when the playhead moves); **Move real to sim** (sends the real gimbal to the Sim gimbal's angles, taking *Move time*; needs the real gimbal reading and Drive on), with a line saying how far the real gimbal is from the sim one; **Copy real pose to sim** (the angles read become the target); a **Look At** component (**+** at the bottom); blue **+** next to **Frames** (insert a frame with the target angles at the playhead, replacing one already there; also the blue **+** on the gimbal's timeline row); **Delete frame** (at the playhead); the frame list (click a frame to move the playhead to it, × deletes it); red **Angle read failed: …** text when reading fails. |
| **Real gimbal** | The angles read, and **Use as target**. |

After about 2 s of failed reads the CAN bus is reopened.

**Look At on the Sim gimbal.** Select the Sim gimbal and add **Look At** with **+**. Pick a **Look At Target** (any other scene object by name, the **Sim cursor**, or a camera script's **Primary target**): while it is on, the Sim gimbal's target aims the lens at it, live (roll 0; pitch is held to the gimbal's range, and straight above or below the mount yaw keeps its last value). The target fields are locked while it aims; the frame **+** keys the aimed angles and **Move real to sim** sends them. During **Run** with Look At on, the real gimbal follows the aim live in short moves instead of the frames. Off or 🗑 gives the target fields back.

**Timeline.** Adding a gimbal adds its track, named by the gimbal's place in the scene (**Gimbal**, **Gimbal 1**, …): teal diamonds are frames, and the gimbal moves linearly from one frame to the next. The track goes with the gimbal object (delete the object to remove it). Changes to the gimbal's settings, target and frames are undo steps (Ctrl+Z), as are pose edits of objects in Properties. During **Simulate** and when scrubbing, the target (and the Sim gimbal) follows the track. During **Run**, the gimbal is driven along the track in one way for the whole run: J1 follows the track with speed commands (resent every tick, closing on the J1 reading, up to 180°/s, so any number of turns), and J2 / J3 are sent the next frame with a move time that makes them arrive on time (gaps longer than 25 s are split). For this the driver puts the RS 4 in its 3D ROLL 360 mode at the start of the run (where its yaw speed turns J1 only) and back to normal at the end, on pause and on stop; the direction of the speed command is learnt from J1's readings (reversed once if J1 turns the wrong way). Not yet tried on the gimbal: the RS 4 tests showed it can swing J1 back by itself when the mode changes with J1 far from 0, and how its J2 / J3 moves behave in that mode is still to be checked. A driver without `track_begin / track_speed / track_end` gets absolute moves for all three joints, so J1 can't turn more than 180° per frame. Before the run, while the arm moves to its init pose, the gimbal goes to its initial pose. Simulate never moves the real gimbal.

**Full turns.** J1 isn't limited to ±180 on the timeline: a frame with J1 360° further on than the one before is one full turn (frames J1 0 → 360 → 720 turn it twice), and the gimbal goes round linearly like any other move between frames. For a roll around the lens, set J3 so the camera looks along the handle and give the next frame J1 + 360. The Target's J1 field takes any value; frames inserted from the real gimbal's reading (orange +) or from Look At / World mode take the J1 nearest the track there, not turns back. (Older project files with a per-frame *spin* setting load as J1 that many turns further on.)

**Gimbal-only timeline.** With no arm path (fewer than 2 waypoints on every arm) and a gimbal track with 2 frames or more, the timeline plays on the gimbal frames alone, up to the last frame; Bake isn't needed. **Simulate** plays them (the Sim gimbal follows, the playhead moves; Pause, Stop, Repeat and Delay work as for the arm). **Upload** sends every gimbal with frames to its initial pose (shown as *Uploading…* for the *Move time*) and needs a real gimbal reading but not the arm; then **Run** drives the real gimbal along the frames, and Repeat sends it back to the initial pose before each new play. Pausing a run stops the clock; the gimbal finishes the move it was sent.

**Driver script.** `user_files/gimbal_scripts/rs4_can.py` is the RS4 reference script (`rs4_can_demo.py`: its `DjiGimbal` class and demo unchanged, except that `move_abs` can skip its wait), with four functions added at the end for the sandbox. It talks to a DJI RS4 through the DJI R SDK (v2.5) over CAN: a CANalyst-II USB adapter by default, 1 Mbit/s, CAN ids 0x223 (to the gimbal) / 0x222 (from it). It needs `pip install python-can pyusb libusb-package`. `docs/examples/gimbal/rs4_spin_360.py` rolls the camera a measured amount around its lens with only J1 turning, for a handle pointing at the target (J3 at about −90°): 3D ROLL 360 mode and the SDK's **yaw** command (in that mode it turns only J1; the roll command pans the camera with all three joints, and in normal mode the roll is held to about ±30°). Default 360° (`--turn -360` the other way; on the test gimbal +J1 looked anticlockwise). The default speed method is smooth: angle requests don't wait for replies, so a speed command goes out every 50 ms, and it ramps up, cruises and slows down onto the target J1 angle (unwrapped across ±180). `--home` turns J1 to 0 first (with J1 far from 0 the gimbal swung back on its own when the mode changed). `--method incr` uses incremental steps (at most 90°, at least 8°: tiny steps are ignored), which pause between steps. It stops with a message if J1 stops following. `docs/examples/gimbal/rs4_axis_probe.py` shows which command moves which joints (absolute yaw only with `--with-abs`: in 3D ROLL 360 it flipped the camera upside down). Run it with the sandbox closed (one program at a time on the CAN adapter). `docs/examples/gimbal/rs4_spin_test.py` spins yaw a full circle with speed commands and prints whether the yaw reading wraps at ±180, how far it really turned and which way (to check multi-turn behaviour on the hardware). Run the driver on its own (`python user_files/gimbal_scripts/rs4_can.py`) to run the reference demo (read the angles, spin yaw 360°, level the pitch). If it stops with *'import can' did not load python-can*, python-can is not installed in that Python and a folder or file named `can` is in the way; the message names both. Another gimbal or adapter can use its own script with the same four functions: `connect(cfg)`, `get_angles()` → (yaw, roll, pitch) in degrees, `move_to(yaw, roll, pitch, duration)` (returns at once) and `close()`. The sandbox calls them from one thread only. The code is `backend/robot_sim/gimbal.py`.

---

## Miscellaneous features

- **File save and load**, with **autosave and automatic backup**.
- **Settings menu** for robot type and joint limits. *Robot Settings* holds the robot IP and port for one arm, chosen in its **Aubo Arm** select (it follows the active arm both ways). The first arm falls back to the address in `arcscontrol.py`, so scripts run on their own keep working; an added arm is sim only until it gets an IP. Robot type, IP, port and the joint speed and acceleration limits are per arm (all on the *Robot Settings* tab, under its **Aubo Arm** select); the first arm's are kept in `settings.json`, added arms' in the project file.
- **Automatic reconnection** to the robot.
- **Multi-camera handling** — the simulator, computer webcams and Sony cameras are handled separately, with split-screen views and an individual recording track for each.
- **View console** for debugging.
- **Path window** — render only a range of waypoints in the 3D viewer at a time, to keep busy scenes readable.
- **View cube** — displays and controls the camera's viewing angle, as in CAD software.
- **Dev mode toggle** with a password lock, to hide development features from regular users.

---

## Game controller (Xbox / PlayStation)

The sandbox can jog the arm with an Xbox or PlayStation (4 / 5) controller connected to the PC it runs on, the same way as the Steam Deck edition. Nothing to set up: the first controller found is used, and it can be plugged in or turned on at any time. The controls are the Deck's, with the shoulder buttons in place of the Deck's back buttons:

| Control | Xbox | PlayStation | Action |
|---|---|---|---|
| Left stick | | | up = +x, left = +y |
| L1 / L2 | LB / LT | L1 / L2 | +z / -z |
| Right stick | | | left = +roll (about x), up = +pitch (about y) |
| R1 / R2 | RB / RT | R1 / R2 | +yaw / -yaw (about z) |
| X | X | Square | switch the controller between the Sim bot and the Real bot |

The top right of the 3D view shows a **Controller** Sim / Real switch (the same as the X button) and the controller in use, for example *Xbox Wireless Controller · Xbox Series X|S* or *PS4 Controller · PS4 (DualShock 4 v2)*, or a warning (no controller, no IK solution, robot not connected). **Preferences > Controller** names the device (with its USB ids), lists the buttons with that controller's own labels, and holds the speeds, shared with the Steam Deck edition: **Stick speed**, **Rotation speed**, **Real ramp time** and **Real jog command**. Sim jogging moves the sim cursor along Axes Mode; Real jogging moves the real arm directly, exactly as on the Deck (see below).

Controllers are read through SDL (the `pygame` package, in `requirements.txt`), which knows Xbox, PlayStation and most other pads and maps them all to the same button positions. Without pygame only Xbox (XInput) controllers work. The code is `backend/gamepad.py` (device, mapping) and `start_controller()` in `backend/deck_webui.py` (the Deck's jog).

**Connecting the controller to the PC**

- **USB cable:** works with any Xbox One / Series or PS4 / PS5 controller. Use a data cable (some charge-only cables do not work).
- **Bluetooth:** Xbox controllers with Bluetooth (Xbox One S and later, and every Xbox Series controller: those have a USB-C port) pair like any Bluetooth device. Hold the small pair button on top until the Xbox button flashes fast, then add it in Windows Settings > Bluetooth. PS4: hold Share + PS until the light bar flashes; PS5: Create + PS.
- **Xbox Wireless Adapter for Windows (the dongle):** controllers are not tied to one adapter; any Xbox One / Series controller can be paired to a new adapter (press the adapter's button, then the controller's pair button). A used controller re-pairs the same way, which also replaces whatever it was paired with before. A controller paired to the adapter or an Xbox console again needs a new pairing for Bluetooth, and the other way round.


A stripped-down controller for running the real arm from a Steam Deck on Windows. Start it with `python backend/deck_webui.py` instead of `webui.py` (it runs the same backend, so don't run both at once).

### ArcsDeck.exe (packaged build)

The Deck edition also ships as one Windows file, **ArcsDeck.exe**, with Python, every package, the robot models, the page and default settings inside it; nothing else needs installing. On start it shows the preloader (`backend/deck/preloader.png`) with the loading log in black on the left of its white strip, for at least 4 seconds, then opens the Deck page in its own window (Edge WebView2, part of Windows 10 / 11). Closing the window quits the app and gives the Steam Deck its previous controller profile back. `ArcsDeck.exe --console` also opens a console window with the full log (`real jog:` lines and so on).

Nothing is saved between runs: the .exe unpacks to a temporary folder that is deleted when it exits, so Settings changes (robot IP, speeds) last for that run only. Defaults, including the Saved Points list, come from `user_files/settings.json` at build time.

To build it:

- **On GitHub:** every push to a `steamdeck/...` branch that touches the app runs the *Build ArcsDeck.exe* workflow (it can also be started by hand from the Actions tab); download `ArcsDeck` from the run's Artifacts.
- **On a Windows PC:** install Python 3.12 from python.org (pyaubo-sdk has no wheels for newer versions) and run `packaging\deck\build_deck.bat`. It makes a build venv (`.venv-deck-build`) and writes `dist\ArcsDeck.exe`.

`python backend/deck_app.py` runs the same launcher from source (preloader and app window; needs `pip install pywebview pillow`). The packaging lives in `packaging/deck/` (`ArcsDeck.spec` for PyInstaller, `requirements-build.txt`, `build_deck.bat`).

The page shows only the 3D viewer, the **Saved Points** list on the left (click one to move the sim cursor there), the **Hold to move** button bottom left (the real arm goes to the sim pose while it is held, and keeps following the sticks), and **Object Properties** on the right with the IK pose and joints of the sim cursor (editable) and the actual robot (read only, shown while connected). The top bar has **Axes Mode** (Local / Global) and **Settings** (robot IP, port, type, Hold to move speed, stick speed, rotation speed, real ramp time and the real jog command). There are no waypoints, bake, timeline, added objects or components, and nothing is autosaved.

| Deck control | Action |
|---|---|
| Left stick | up = +x, left = +y (x / y of Axes Mode) |
| L4 / L5 | +z / -z |
| Right stick | left = +roll (about x), up = +pitch (about y) |
| R4 / R5 | +yaw / -yaw (about z) |
| X | switch the controller between the Sim bot and the Real bot (also the **Control** switch in the top bar) |
| Right trackpad | mouse (drag the sim cursor gizmo as in the sandbox) |
| R2 / L2 (or right pad click) | left / right click |
| Left trackpad | scroll (zoom) |

In **Real** control the sticks jog the real arm directly. Each jog first calls `RobotArcs.PrepareJog` (`startMove`, so a paused motion queue does not swallow the jog, servo mode off, speed fraction 100 %). The **Real jog command** setting picks `servoJoint` (servo mode, on its own thread at a fixed 20 ms with 1 ms Windows timers: each tick moves a target TCP pose one period along the ramped stick speed and sends the controller's IK of it with t = 20 ms; the target is held only if it gets 8 cm ahead of the arm), `speedLine` (TCP velocity) or `speedJoint` (joint velocities toward the controller's IK of a pose 0.25 s ahead). Each speed command restarts the controller's move, so speedLine and speedJoint are re-sent only when the sticks change by a tenth of full tilt or every 0.8 s (1 s timeout, `StopSpeed` on release). servoJoint is the default. Real jogs accelerate and decelerate: **Real ramp time** (default 0.5 s) is the time from stop to full stick speed, and the same rate eases the arm to a stop when you let go (servoJoint keeps streaming until the speed reaches zero; speedLine / speedJoint pass the matching acceleration to the controller). If the joints have not moved after about 3 cm of commanded travel the page says so, and a jog is refused while the robot is not Running or its safety mode is not Normal (for example a protective stop, which is cleared on the teach pendant). Return codes and the controller state (robot, safety and operational mode, runtime state) are printed as `real jog:` lines in the console; the sim bot snaps to the real arm when you switch to Real and when you let go. Real control needs a connected robot and drops back to Sim if it disconnects. Moves follow Axes Mode, which starts on Global (the robot base x / y / z). Local uses the tool's own axes: with the tool pointing down these are turned (tool x is base y and tool z is down), so the sticks seem swapped. While jogging, a `real jog:` line each second shows the sticks, the base-frame speed sent and, for servoJoint, how often the target waited for the arm or had no IK. Speeds are the *Stick speed* (default 0.1 m/s at full tilt, linear stick response) and *Rotation speed* settings, sim and real; a move with no IK solution is dropped and the panel says so.

The controls come from [Steam Deck Tools](https://github.com/WaterloggedRobots/steam-deck-tools)' `SteamController.exe`, which needs its hidden **Arcs Deck** profile (a build from that repository that includes it). On start the app switches `SteamController.exe` to Arcs Deck and switches back to the previous profile on exit; if the app is killed, the profile turns itself off. The Deck reaches the app as an Xbox 360 pad over XInput (`backend/deck/pad.py`); the profile switch is in `backend/deck/steam_profile.py`.

If the controls are not on the Arcs Deck profile, a red banner over the 3D view says why and what still works: **Steam Deck Tools is not running** (sticks and back buttons do nothing; the app switches over by itself once Steam Controller starts), **wrong Steam Deck Tools build** (no Arcs Deck profile: in X360 the sticks still jog but up / down and yaw sit on L1 / R1 and L2 / R2, in Desktop the sticks do nothing), or **the profile was changed** while the app runs. **Retry** tries the switch again; the cross hides the banner until the problem changes. The profile is checked every 3 seconds.

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
| Gimbal | Gimbal frames can't be dragged on the timeline (delete and insert again). **Move real to sim**, the initial pose and Run send the target with the SDK's absolute move, which the RS4 tests showed is taken as the camera's *world* angles, not J1–J3: with the gimbal mounted sideways (handle horizontal) the real gimbal goes to the wrong pose. Turn **Drive** off for such mounts until joint control is added. J2's sign and J1's direction as seen on the twins are not yet checked against the hardware. |
| Several Aubo Arms | Undo and redo belong to the active arm. Timeline extra tracks are shared by all arms; the Look At track shows the active arm only. |

---

## Project layout

Run the app with `python backend/webui.py`. Every file the app reads or writes is resolved in `backend/app_paths.py`, so it works from any working directory. The tree is split three ways with a future Electron build in mind:

```
backend/                  everything needed to run the app
  webui.py                Flask web UI (entry point)
  deck_webui.py           Steam Deck edition (entry point), templates/deck.html
  deck_app.py             Steam Deck edition as a desktop app (ArcsDeck.exe entry point: preloader + window)
  gamepad.py              Xbox / PlayStation controller for the sandbox (SDL via pygame, XInput fallback)
  deck/                   Steam Deck controls (XInput) and Steam Deck Tools profile switch
  app_paths.py            folder constants
  templates/index.html    the page
  cameras/                sony_liveview.py, libusb-1.0.dll (Sony camera support)
  state/temp.json         autosave (internal, not for hand editing)
  robot_sim/
    robot_sim_run.py      Viser viewer loop and request queues
    robot_runtime.py      sim/real robot state, IK, Simulate / Run, path buffer
    waypoints.py          waypoints, Bezier handles, bookmarked points, saved paths, selection, undo/redo
    simulatorview.py      other scene objects: primitives, meshes, parenting, measure, cameras
    gimbal.py             Gimbal objects: Sim / Real twins, gimbal track, driver script thread (DJI RS4)
    gimbal_model.py       gimbal model files: loading, kinematics, Look At aim, box models
    path_bake.py          path sampling, IK and speed checks, baking
    timeline.py           timeline control: keyframe drag/snap, sample times, transitions, copy/paste
    file_manage.py        save / open / export
    arcscontrol.py        AUBO SDK wrapper
    aubo-i5-full-master/  robot URDFs and meshes
user_files/               ships with the app, editable by the user
  settings.json           settings, global saved points and paths
  cam_scripts/            OpenCV / RealSense scripts added from the GUI
  gimbal_scripts/         gimbal driver scripts (rs4_can.py: DJI RS4 over CAN)
  gimbal_models/          gimbal models: joint axes, offsets, boxes (dji_rs4.json)
  paths/                  saved .arcspath projects
  exports/                export.json, bake_overspeed.csv / .xlsx
  recordings/             camera recordings
  imported_meshes/        meshes added with Import mesh
packaging/deck/           ArcsDeck.exe build: PyInstaller spec, build requirements, build_deck.bat
docs/                     not needed at runtime
  examples/cam_scripts/   example camera scripts (YOLO / RealSense), models/ holds yolov8n.pt
  examples/gimbal/        rs4_axis_probe.py: which RS4 command turns one joint only; rs4_spin_360.py: RS4 closed-loop camera roll (measured 360°, 3D ROLL 360); rs4_spin_test.py: diagnostics (wrap, direction, resend timing)
  CameraRemoteCommand-2.00.02/   Sony PTP SDK reference and examples
  archive/, demo/, export-format/, Function List, slides, SDK manual, bug log
```

A packaged build can move the writable folders with the `ARCS_USER_DIR` (for `user_files`) and `ARCS_STATE_DIR` (for `backend/state`) environment variables.

Functions moved into `simulatorview.py` and `timeline.py` are still reachable through their old module (for example `wp.add_scene_object` or `bake.pin_key_times`), so existing calls keep working.
