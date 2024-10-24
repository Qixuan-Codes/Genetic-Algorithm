import pybullet as p
import math
import random
import os
import numpy as np

class Simulation:
    def __init__(self, sim_id=0):
        # Initialise the simulation and connect to PyBullet in DIRECT mode
        self.physicsClientId = p.connect(p.DIRECT)
        self.sim_id = sim_id

    def make_mountain(self, num_rocks=0, max_size=0.25, arena_size=10, mountain_height=5):
        def gaussian(x, y, sigma=arena_size/4):
            """Return the height of the mountain at position (x, y) using a Gaussian function."""
            return mountain_height * math.exp(-((x**2 + y**2) / (2 * sigma**2)))

        for _ in range(num_rocks):
            x = random.uniform(-arena_size/2, arena_size/2)
            y = random.uniform(-arena_size/2, arena_size/2)
            z = gaussian(x, y)  # Height determined by the Gaussian function

            size_factor = 1 - (z / mountain_height)
            size = random.uniform(0.1, max_size) * size_factor

            orientation = p.getQuaternionFromEuler([random.uniform(0, 3.14), random.uniform(0, 3.14), random.uniform(0, 3.14)])
            rock_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size, size, size])
            rock_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[size, size, size], rgbaColor=[0.5, 0.5, 0.5, 1])
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=rock_shape, baseVisualShapeIndex=rock_visual, basePosition=[x, y, z], baseOrientation=orientation)

    def make_arena(self, arena_size=10, wall_height=1):
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

    def sense_environment(self, cid, target_coords):
        # Sense the environment and calculate the distance from the creature to the target
        pos, _ = p.getBasePositionAndOrientation(cid, physicsClientId=self.physicsClientId)
        distance_to_target = np.linalg.norm(np.array(target_coords) - np.array(pos))
        return distance_to_target

    def run_creature(self, cr, iterations=None, target_coords=None):
        # Run the simulation for a given creature
        pid = self.physicsClientId
        p.resetSimulation(physicsClientId=pid)
        p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=pid)

        p.setGravity(0, 0, -10, physicsClientId=pid)
        self.make_arena(arena_size=20, wall_height=1)
        self.make_mountain(num_rocks=0, max_size=0.25, arena_size=20, mountain_height=5)

        mountain_position = (0, 0, -1)
        mountain_orientation = p.getQuaternionFromEuler((0, 0, 0))
        urdf_path = os.path.join(os.path.dirname(__file__), 'shapes', 'gaussian_pyramid.urdf')
        mountain = p.loadURDF(urdf_path, mountain_position, mountain_orientation, useFixedBase=1, physicsClientId=pid)

        # Save the creature's URDF to a temp file
        xml_file = 'temp' + str(self.sim_id) + '.urdf'
        xml_str = cr.to_xml()
        with open(xml_file, 'w') as f:
            f.write(xml_str)
        
        # Load the creature into the simulation
        cid = p.loadURDF(xml_file, physicsClientId=pid)

        # Reset the creature's position and orientation
        p.resetBasePositionAndOrientation(cid, [-9, 0, 2.5], [0, 0, 0, 1], physicsClientId=pid)

        for step in range(iterations):
            p.stepSimulation(physicsClientId=pid)
            # if step % 24 == 0:
                # self.update_motors(cid=cid, cr=cr, target_coords=target_coords)

            pos, orn = p.getBasePositionAndOrientation(cid, physicsClientId=pid)
            cr.update_position(pos)
        
        # Calculate the distance based on the creature's position and the peak of the mountain
        if target_coords is not None:
            cr.fitness = -cr.distance_to_target(target_coords)
        else:
            cr.fitness = cr.get_distance_travelled()
        return cr.fitness
    
    def update_motors(self, cid, cr, target_coords):
        # Update the motors of the creature to control its movement
        kp = 0.7
        kd = 0.15

        for jid in range(p.getNumJoints(cid, physicsClientId=self.physicsClientId)):
            m = cr.get_motors()[jid]
            joint_info = p.getJointInfo(cid, jid, physicsClientId=self.physicsClientId)
            joint_position = p.getLinkState(cid, joint_info[0], physicsClientId=self.physicsClientId)[0]
            joint_orientation = p.getLinkState(cid, joint_info[0], physicsClientId=self.physicsClientId)[1]

            direction_to_target = np.array(target_coords) - np.array(joint_position)
            distance_to_target = np.linalg.norm(direction_to_target)
            direction_to_target /= distance_to_target

            # Assuming the creature is facing along the x-axis of its local coordinate system
            facing_direction = p.getMatrixFromQuaternion(joint_orientation)
            facing_direction = np.array([facing_direction[0], facing_direction[3], facing_direction[6]])  # Get the x-axis vector

            # Calculate the alignment with the target direction
            alignment = np.dot(direction_to_target, facing_direction)

            # Set the boost to be higher as the creature get closer to the target coordinates (peak of the mountain)
            boost = 50.0 / (distance_to_target + 1e-5)

            # Adjusted motor output based on alignment and boost
            adjusted_velocity = m.get_output() * (1 + alignment) * boost

            # PD Control
            current_vel = p.getJointState(cid, jid, physicsClientId=self.physicsClientId)[1]
            desired_vel = adjusted_velocity
            error = desired_vel - current_vel
            control_signal = kp * error - kd * current_vel

            # Set the motor control
            p.setJointMotorControl2(cid, jid, 
                    controlMode=p.VELOCITY_CONTROL, 
                    targetVelocity=control_signal, 
                    force=200,
                    physicsClientId=self.physicsClientId)
