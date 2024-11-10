import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
import pybullet as p

class BatteryWayPointAviary(BaseRLAviary):
    """Single agent RL problem: hover at position."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM,
                 battery_alpha: float = 0.0
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)
        battery_alpha : float, optional
            The weight of battery penalty in the reward function

        """
        self.TARGET_POS = np.array([0,0,1])
        self.EPISODE_LEN_SEC = 16
        self.FIXED_Z = 1.0  # Fixed Z-height for all waypoints
        self.NUM_WAYPOINTS = 10  # Number of intermediate waypoints
        self.EPSILON = 0.1  # Distance threshold to consider waypoint reached
        self.TIME_LIMIT_SEC = 20.0  # Time limit in seconds
        self.FINAL_DISTANCE = 2.0  # Final x-coordinate destination
        self.TERMINATION_BOUND = 5.0  # Maximum allowed distance from target
        self.battery_alpha = battery_alpha
        self.current_distance_reward = 0.0
        self.current_battery_penalty = 0.0
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )

    ################################################################################
    
    def reset(self,
              seed : int = None,
              options : dict = None):
        """Resets the environment.

        Parameters
        ----------
        seed : int, optional
        """
        # Generate waypoints along x-axis with fixed Z
        self.waypoints = []
        self.waypoints.append(np.array([0, 0, self.FIXED_Z]))  # First waypoint
        
        # Generate intermediate waypoints
        for m in range(1, self.NUM_WAYPOINTS):
            self.waypoints.append(np.array([m/10, 0, self.FIXED_Z]))
        
        # Add final waypoint
        self.waypoints.append(np.array([self.NUM_WAYPOINTS/10, 0, self.FIXED_Z]))
        
        self.current_waypoint_idx = 0
        self.waypoints = np.array(self.waypoints)
        
        initial_obs, initial_info = super().reset()
        return initial_obs, initial_info
        
    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)
        current_pos = state[0:3]
        target_pos = self.waypoints[self.current_waypoint_idx]
        
        # Check if we reached current waypoint
        if np.linalg.norm(current_pos - target_pos) < self.EPSILON:
            if self.current_waypoint_idx < len(self.waypoints) - 1:
                self.current_waypoint_idx += 1
        
        # Calculate distance reward
        self.current_distance_reward = -np.mean((current_pos - target_pos)**2)
        
        # Calculate battery penalty
        self.current_battery_penalty = 0
        if self.battery_alpha > 0:
            rpm = self._getDroneStateVector(0)[10:14]
            self.current_battery_penalty = -np.mean(rpm**2) / (self.MAX_RPM**2)
        
        # Calculate total reward
        total_reward = self.current_distance_reward + (self.battery_alpha * self.current_battery_penalty)
        
        return total_reward

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        return self._computeTruncated()
        

        
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        state = self._getDroneStateVector(0)
        current_pos = state[0:3]
        target_pos = self.waypoints[self.current_waypoint_idx]
        
        # Check time limit
        if self.step_counter/self.PYB_FREQ > self.TIME_LIMIT_SEC:
            return True
        
        # Check distance from current target waypoint
        if np.linalg.norm(current_pos - target_pos) > self.TERMINATION_BOUND:
            return True
        
        # # Check other truncation conditions
        # if (abs(state[7]) > .4 or abs(state[8]) > .4):  # Truncate when the drone is too tilted
        #     return True
        if self.current_waypoint_idx >= len(self.waypoints) - 1:
            return True 
        
        return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Returns
        -------
        dict[str, float]
            Information about the current reward components

        """
        return {
            "distance_reward": self.current_distance_reward,
            "battery_penalty": self.current_battery_penalty,
            "battery_weighted_penalty": self.battery_alpha * self.current_battery_penalty
        }

    # def step(self,
    #          action
    #          ):
    #     """Advances the environment by one simulation step.

    #     Parameters
    #     ----------
    #     action : ndarray | dict[..]
    #         The input action for one or more drones, translated into RPMs by
    #         the specific implementation of `_preprocessAction()` in each subclass.

    #     Returns
    #     -------
    #     ndarray | dict[..]
    #         The step's observation, check the specific implementation of `_computeObs()`
    #         in each subclass for its format.
    #     float | dict[..]
    #         The step's reward value(s), check the specific implementation of `_computeReward()`
    #         in each subclass for its format.
    #     bool | dict[..]
    #         Whether the current episode is over, check the specific implementation of `_computeTerminated()`
    #         in each subclass for its format.
    #     bool | dict[..]
    #         Whether the current episode is truncated, check the specific implementation of `_computeTruncated()`
    #         in each subclass for its format.
    #     bool | dict[..]
    #         Whether the current episode is trunacted, always false.
    #     dict[..]
    #         Additional information as a dictionary, check the specific implementation of `_computeInfo()`
    #         in each subclass for its format.

    #     """
    #     #### Save PNG video frames if RECORD=True and GUI=False ####
    #     if self.RECORD and not self.GUI and self.step_counter%self.CAPTURE_FREQ == 0:
    #         [w, h, rgb, dep, seg] = p.getCameraImage(width=self.VID_WIDTH,
    #                                                  height=self.VID_HEIGHT,
    #                                                  shadow=1,
    #                                                  viewMatrix=self.CAM_VIEW,
    #                                                  projectionMatrix=self.CAM_PRO,
    #                                                  renderer=p.ER_TINY_RENDERER,
    #                                                  flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
    #                                                  physicsClientId=self.CLIENT
    #                                                  )
    #         (Image.fromarray(np.reshape(rgb, (h, w, 4)), 'RGBA')).save(os.path.join(self.IMG_PATH, "frame_"+str(self.FRAME_NUM)+".png"))
    #         #### Save the depth or segmentation view instead #######
    #         # dep = ((dep-np.min(dep)) * 255 / (np.max(dep)-np.min(dep))).astype('uint8')
    #         # (Image.fromarray(np.reshape(dep, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
    #         # seg = ((seg-np.min(seg)) * 255 / (np.max(seg)-np.min(seg))).astype('uint8')
    #         # (Image.fromarray(np.reshape(seg, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
    #         self.FRAME_NUM += 1
    #         if self.VISION_ATTR:
    #             for i in range(self.NUM_DRONES):
    #                 self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i)
    #                 #### Printing observation to PNG frames example ############
    #                 self._exportImage(img_type=ImageType.RGB, # ImageType.BW, ImageType.DEP, ImageType.SEG
    #                                 img_input=self.rgb[i],
    #                                 path=self.ONBOARD_IMG_PATH+"/drone_"+str(i)+"/",
    #                                 frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
    #                                 )
    #     #### Read the GUI's input parameters #######################
    #     if self.GUI and self.USER_DEBUG:
    #         current_input_switch = p.readUserDebugParameter(self.INPUT_SWITCH, physicsClientId=self.CLIENT)
    #         if current_input_switch > self.last_input_switch:
    #             self.last_input_switch = current_input_switch
    #             self.USE_GUI_RPM = True if self.USE_GUI_RPM == False else False
    #     if self.USE_GUI_RPM:
    #         for i in range(4):
    #             self.gui_input[i] = p.readUserDebugParameter(int(self.SLIDERS[i]), physicsClientId=self.CLIENT)
    #         clipped_action = np.tile(self.gui_input, (self.NUM_DRONES, 1))
    #         if self.step_counter%(self.PYB_FREQ/2) == 0:
    #             self.GUI_INPUT_TEXT = [p.addUserDebugText("Using GUI RPM",
    #                                                       textPosition=[0, 0, 0],
    #                                                       textColorRGB=[1, 0, 0],
    #                                                       lifeTime=1,
    #                                                       textSize=2,
    #                                                       parentObjectUniqueId=self.DRONE_IDS[i],
    #                                                       parentLinkIndex=-1,
    #                                                       replaceItemUniqueId=int(self.GUI_INPUT_TEXT[i]),
    #                                                       physicsClientId=self.CLIENT
    #                                                       ) for i in range(self.NUM_DRONES)]
    #     #### Save, preprocess, and clip the action to the max. RPM #
    #     else:
    #         clipped_action = np.reshape(self._preprocessAction(action), (self.NUM_DRONES, 4))
    #     #### Repeat for as many as the aggregate physics steps #####
    #     for _ in range(self.PYB_STEPS_PER_CTRL):
    #         #### Update and store the drones kinematic info for certain
    #         #### Between aggregate steps for certain types of update ###
    #         if self.PYB_STEPS_PER_CTRL > 1 and self.PHYSICS in [Physics.DYN, Physics.PYB_GND, Physics.PYB_DRAG, Physics.PYB_DW, Physics.PYB_GND_DRAG_DW]:
    #             self._updateAndStoreKinematicInformation()
    #         #### Step the simulation using the desired physics update ##
    #         for i in range (self.NUM_DRONES):
    #             if self.PHYSICS == Physics.PYB:
    #                 self._physics(clipped_action[i, :], i)
    #             elif self.PHYSICS == Physics.DYN:
    #                 self._dynamics(clipped_action[i, :], i)
    #             elif self.PHYSICS == Physics.PYB_GND:
    #                 self._physics(clipped_action[i, :], i)
    #                 self._groundEffect(clipped_action[i, :], i)
    #             elif self.PHYSICS == Physics.PYB_DRAG:
    #                 self._physics(clipped_action[i, :], i)
    #                 self._drag(self.last_clipped_action[i, :], i)
    #             elif self.PHYSICS == Physics.PYB_DW:
    #                 self._physics(clipped_action[i, :], i)
    #                 self._downwash(i)
    #             elif self.PHYSICS == Physics.PYB_GND_DRAG_DW:
    #                 self._physics(clipped_action[i, :], i)
    #                 self._groundEffect(clipped_action[i, :], i)
    #                 self._drag(self.last_clipped_action[i, :], i)
    #                 self._downwash(i)
    #         #### PyBullet computes the new state, unless Physics.DYN ###
    #         if self.PHYSICS != Physics.DYN:
    #             p.stepSimulation(physicsClientId=self.CLIENT)
    #         #### Save the last applied action (e.g. to compute drag) ###
    #         self.last_clipped_action = clipped_action
    #     #### Update and store the drones kinematic information #####
    #     self._updateAndStoreKinematicInformation()
    #     #### Prepare the return values #############################
    #     obs = self._computeObs()
    #     reward = self._computeReward()
    #     terminated = self._computeTerminated()
    #     truncated = self._computeTruncated()
    #     info = self._computeInfo()
    #     #### Advance the step counter ##############################
    #     self.step_counter = self.step_counter + (1 * self.PYB_STEPS_PER_CTRL)
    #     return obs, reward, terminated, truncated, info
    