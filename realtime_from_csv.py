import os
import pybullet as p
import time
import numpy as np
import genome
import creature
import random

# Change the file location before running
CSV_FILE = "assets/Parameter2-Exceptional/fittest_creature.csv"
URDF_FILE = "assets/Parameter2-Exceptional/fittest_creature.urdf"

# CSV_FILE = "assets/Parameter2-Basic/fittest_creature.csv"
# URDF_FILE = "assets/Parameter2-Basic/fittest_creature.urdf"

def make_mountain(num_rocks=0, max_size=0.25, arena_size=10, mountain_height=5):
    def gaussian(x, y, sigma=arena_size/4):
        """Return the height of the mountain at position (x, y) using a Gaussian function."""
        return mountain_height * math.exp(-((x**2 + y**2) / (2 * sigma**2)))

    for _ in range(num_rocks):
        x = random.uniform(-arena_size/2, arena_size/2)
        y = random.uniform(-arena_size/2, arena_size/2)
        z = gaussian(x, y)

        size_factor = 1 - (z / mountain_height)
        size = random.uniform(0.1, max_size) * size_factor

        orientation = p.getQuaternionFromEuler([random.uniform(0, 3.14), random.uniform(0, 3.14), random.uniform(0, 3.14)])
        rock_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size, size, size])
        rock_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[size, size, size], rgbaColor=[0.5, 0.5, 0.5, 1])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=rock_shape, baseVisualShapeIndex=rock_visual, basePosition=[x, y, z], baseOrientation=orientation)

def make_arena(arena_size=10, wall_height=1):
    wall_thickness = 0.5
    floor_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size/2, arena_size/2, wall_thickness])
    floor_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size/2, arena_size/2, wall_thickness], rgbaColor=[1, 1, 0, 1])
    floor_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=floor_collision_shape, baseVisualShapeIndex=floor_visual_shape, basePosition=[0, 0, -wall_thickness])

    wall_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size/2, wall_thickness/2, wall_height/2])
    wall_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size/2, wall_thickness/2, wall_height/2], rgbaColor=[0.7, 0.7, 0.7, 1])  # Gray walls

    # Create four walls
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[0, arena_size/2, wall_height/2])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[0, -arena_size/2, wall_height/2])

    wall_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[wall_thickness/2, arena_size/2, wall_height/2])
    wall_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[wall_thickness/2, arena_size/2, wall_height/2], rgbaColor=[0.7, 0.7, 0.7, 1])  # Gray walls

    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[arena_size/2, 0, wall_height/2])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[-arena_size/2, 0, wall_height/2])

def main():
    # Check if file exists
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} does not exist.")
        return

    if not os.path.exists(URDF_FILE):
        print(f"Error: {URDF_FILE} does not exist.")
        return

    p.connect(p.GUI)
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    make_arena(arena_size=20, wall_height=2)
    make_mountain(num_rocks=0, max_size=0.25, arena_size=10, mountain_height=5)

    # Adding the mountain URDF
    mountain_position = (0, 0, -1)  # Adjust as needed
    mountain_orientation = p.getQuaternionFromEuler((0, 0, 0))
    p.setAdditionalSearchPath('shapes/')
    p.loadURDF("gaussian_pyramid.urdf", mountain_position, mountain_orientation, useFixedBase=1)

    p.setGravity(0, 0, -10)

    # Load the fittest creature URDF into the sim
    rob1 = p.loadURDF(URDF_FILE)
    # Air drop it
    p.resetBasePositionAndOrientation(rob1, [-8.5, 0, 2.5], [0, 0, 0, 1])
    start_pos, orn = p.getBasePositionAndOrientation(rob1)

    # Load the DNA from the CSV file
    dna = genome.Genome.from_csv(CSV_FILE)
    flattened_dna = [float(item) for sublist in dna for item in sublist]

    # Target vertex coordinates (the peak of the mountain)
    target_vertex = np.array([0.5, 0.0, 5.07751996])

    # Set camera
    p.resetDebugVisualizerCamera(cameraDistance=17, cameraYaw=-39, cameraPitch=-31, cameraTargetPosition=[0, 0, 0])

    # PD control parameters
    kp = 0.7
    kd = 0.15

    # Iterate 
    elapsed_time = 0
    wait_time = 1.0 / 240  # seconds
    total_time = 30  # seconds
    step = 0
    max_height = -float('inf')  # Track the maximum height

    while True:
        p.stepSimulation()
        step += 1
        if step % 24 == 0:
            for jid in range(p.getNumJoints(rob1)):
                vel = flattened_dna[jid % len(flattened_dna)]  # Get the motor output from the flattened DNA

                # Adjust the motor velocity based on distance to target
                joint_info = p.getJointInfo(rob1, jid)
                joint_position = p.getLinkState(rob1, joint_info[0])[0]  # Get the position of the joint
                joint_orientation = p.getLinkState(rob1, joint_info[0])[1]  # Get the orientation of the joint

                # Calculate the direction to the target
                direction_to_target = target_vertex - np.array(joint_position)
                distance_to_target = np.linalg.norm(direction_to_target)
                direction_to_target /= distance_to_target

                # Assuming the creature is facing along the x-axis of its local coordinate system
                facing_direction = p.getMatrixFromQuaternion(joint_orientation)
                facing_direction = np.array([facing_direction[0], facing_direction[3], facing_direction[6]])  # Get the x-axis vector

                # Calculate the alignment with the target direction
                alignment = np.dot(direction_to_target, facing_direction)

                # Enhanced boost factor based on distance to the target
                boost = 50.0 / (distance_to_target + 1e-5)

                # Adjusted motor output based on alignment and boost
                adjusted_velocity = vel * (1 + alignment) * boost

                # PD Control
                current_vel = p.getJointState(rob1, jid)[1]
                desired_vel = adjusted_velocity
                error = desired_vel - current_vel
                control_signal = kp * error - kd * current_vel

                p.setJointMotorControl2(rob1, jid, 
                        controlMode=p.VELOCITY_CONTROL, 
                        targetVelocity=control_signal, 
                        force=200)

                print(f"Control Signal for Joint {jid}: {control_signal}")
                print(f"Distance to Target: {distance_to_target}")
                print(f"Alignment: {alignment}")

        time.sleep(wait_time)
        elapsed_time += wait_time
        if elapsed_time > total_time:
            break

    # Print the maximum height achieved
    print("DISTANCE TO TARGET VERTEX:", distance_to_target)
    print("MAXIMUM HEIGHT ACHIEVED:", max_height)

if __name__ == "__main__":
    main()
