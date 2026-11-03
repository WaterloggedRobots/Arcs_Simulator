# Steam Deck edition: controls

Controls of the Steam Deck edition (`ArcsDeck.exe` / `python backend/deck_webui.py`), version **deck-1.10**.
They need Steam Deck Tools' `SteamController.exe` with its **Arcs Deck** profile; the app switches to it on start
and back on exit. Without it (red banner over the 3D view) only the touchscreen works fully.

## Buttons and sticks: moving the robot

The sticks and back buttons move the **Sim bot** (sim cursor) or, after **X**, the **Real bot** directly.
Directions follow **Axes Mode** in the top bar (Global = robot base axes, the default; Local = the tool's own axes).

| Control | Action |
|---|---|
| Left stick up / down | +x / -x |
| Left stick left / right | +y / -y |
| L4 / L5 (upper / lower left back button) | +z / -z (up / down) |
| Right stick left / right | +roll / -roll (about x) |
| Right stick up / down | +pitch / -pitch (about y) |
| R4 / R5 (upper / lower right back button) | +yaw / -yaw (about z) |
| X | switch between the **Sim** bot and the **Real** bot (same as the **Control** switch in the top bar) |
| D-pad up / down | zoom the 3D view in / out by 10 % per press (the **Zoom** slider follows) |

Speeds are set in **Settings**: *Stick speed* (m/s at full tilt), *Rotation speed* (°/s), and for the real arm
*Real ramp time* (seconds to reach full speed and to stop) and *Real jog command*.

Not used: A, B, Y, L1, R1, D-pad left / right, Start, Select (View / Menu), stick clicks.

## Trackpads: the mouse cursor

The trackpads and triggers are a normal mouse, handled by Steam Controller itself.

| Control | Action |
|---|---|
| Right trackpad | move the cursor |
| R2 | left click (hold = hold the button down) |
| L2, or press (click) the right trackpad | right click |
| Left trackpad | scroll (zoom in the 3D view) |

What the cursor does on the page:

| Where | Cursor action | Result |
|---|---|---|
| 3D view, on the gizmo (arrows / rings at the sim cursor) | left-drag | move / rotate the sim cursor (the robot follows when IK has a solution) |
| 3D view, empty space | left-drag | turn the view (around the point the view looks at) |
| 3D view | right-drag | slide the view sideways / up and down |
| 3D view | scroll (left trackpad) | zoom (the **Zoom** slider follows) |
| Saved Points (left) | click a point | move the sim cursor to it |
| **Hold to move** (bottom left) | hold the left click | the real arm drives to the sim cursor's pose while held, and keeps following the sticks; let go to stop (greyed out until the robot is connected) |
| Object Properties (right), Sim cursor | click a field, type, Enter | set the sim cursor's IK (x y z m, rx ry rz deg) or joints (deg) |
| Top bar | click | **Control** Sim / Real, **Zoom** slider, **Axes Mode** Local / Global, **Settings** |

## Touchscreen

| Where you touch | Action |
|---|---|
| 3D view, one finger dragged left / right | orbit the view around the arm (around the arm's base) |
| 3D view, one finger dragged up / down | view the arm from higher / lower (stops just short of straight down / up) |
| Anywhere else (panels, buttons, top bar, Settings) | tap = click, as with the cursor (Hold to move: keep the finger on the button) |

Orbiting keeps the zoom; the first drag centres the view on the arm's base. The touch layer over the 3D view
steps aside while the trackpad cursor is over the view (so gizmo drags and scrolling reach the 3D view), and comes
back as soon as the cursor leaves the view or a finger touches the rest of the page. While the cursor rests over
the 3D view, a finger there uses the 3D view's own touch controls instead (one finger turns the view around the
point it looks at, two fingers pinch to zoom).

## Status in the top bar

| Pill | Meaning |
|---|---|
| Robot connected / not connected | the ARCS controller at the IP and port in Settings |
| Controller: Arcs Deck | Steam Controller is on the Arcs Deck profile: sticks and back buttons work |
| Controller connected | a pad is there, but not on the Arcs Deck profile (see the red banner) |
| No controller | no XInput pad: Steam Controller is not running |
