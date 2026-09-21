
import math
import time
import numpy as np
from flask import Flask, request, jsonify
import threading

robot_ip = "192.168.0.101"
robot_port = 30004
current_joints = [0/180*math.pi,0/180*math.pi,90/180*math.pi,0/180*math.pi,90/180*math.pi,0/180*math.pi]

class RobotArcs:
    def __init__(self, ip=robot_ip, port=robot_port, user="aubo", pwd="123456"):
        self.placeholder = None

    def wait_arrival(self, start_timeout_s=2.0):
        return 0

    def MoveLAwait(self, pose_xyzrpy, vel=1.2, acc=0.25, blend_r=0.0, blend_type=0):
        return self.wait_arrival()

    def MoveLAsync(self, pose_xyzrpy, vel=1.2, acc=0.25, blend_r=0.0, blend_type=0):
        return None

    def MoveJAwait(self, jvel=80*math.pi/180, jacc=60*math.pi/180, blend_r=0.0, blend_type=0):    #input as rad
        return self.wait_arrival()

    def GetTcpPose(self):  # base->TCP
        return True

    def GetToolPose(self):  # base->Tool
        return [-1,-1,-1,-1,-1,-1]

    def GetJoints(self):    #joints as rad
        return current_joints

    def StopMove(self):
        return None

    def StartMove(self):
        return None

    def Close(self):
        return None

    def GetIK(self):
        return None

    def SetSpeed(self,new_speed):
        return None

    def estimate_move_duration(self, current_q, target_q, speed_fraction=0.3, max_joint_speed=100):
        """
        Estimate how long a joint move will take on the real robot.

        Parameters
        ----------
        current_q : list or np.array
            Current joint angles (rad)
        target_q : list or np.array
            Target joint angles (rad)
        speed_fraction : float
            Speed percentage used on the real robot (0.0 ~ 1.0)
        max_joint_speed : float
            Maximum joint speed of the robot (rad/s)

        Returns
        -------
        duration : float
            Estimated time in seconds
        """
        current_q = np.array(current_q)
        target_q = np.array(target_q)

        # Largest joint angle change
        max_delta = np.max(np.abs(target_q - current_q))

        # Effective speed

        effective_speed = max_joint_speed * speed_fraction

        # Avoid division by zero
        if effective_speed < 1e-6:
            return 10.0

        duration = max_delta / effective_speed

        # Add a small margin (real robot has acceleration/deceleration)
        duration *= 1.15

        return max(duration, 0.3)  # minimum 0.3

def run_flask():
    # Important: use threaded=True
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)

app = Flask(__name__)

@app.route("/joints", methods=["POST"])
def receive_joints():
    global current_joints
    data = request.get_json(force=True)

    # Expecting: {"joints": [j1, j2, j3, j4, j5, j6]}
    if "joints" not in data:
        return jsonify({"error": "Missing 'joints' field"}), 400

    joints = data["joints"]

    if not isinstance(joints, list) or len(joints) != 6:
        return jsonify({"error": "joints must be a list of exactly 6 numbers"}), 400

    try:
        joints = [float(j) for j in joints]
    except (TypeError, ValueError):
        return jsonify({"error": "All joint values must be numbers"}), 400

    current_joints = joints
    print("Received joints:", joints)

    return jsonify({
        "status": "ok",
        "received": joints
    })

flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()
print("Flask API running on http://0.0.0.0:5000")

if __name__ == '__main__':
    print("Nothing happens here")
