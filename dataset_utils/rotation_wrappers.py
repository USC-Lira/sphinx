from dataclasses import dataclass, field
from collections import defaultdict, namedtuple
import os
from typing import Optional
import numpy as np
from scipy.spatial.transform import Rotation as R

@dataclass
class EulerQuatRotationWrapperConfig:
    prop_dim: int
    rotation_start_indices: list[int]
    from_euler: bool = True  # True if dataset is in Euler, False if in quaternion
    policy_in_euler: bool = False
    eval_in_euler: bool = True

# Use when the dataset and evaluation code uses different rotation formats
class EulerQuatRotationWrapper:
    """Wrapper to convert proprioception and actions between Euler and quaternion representations."""
    
    @classmethod
    def from_config(cls, config: EulerQuatRotationWrapperConfig):
        """Initialize from config."""
        if config.from_euler:
            return cls.from_euler(
                prop_dim=config.prop_dim,
                euler_start_indices=config.rotation_start_indices,
                dataset_in_euler=config.from_euler,
                policy_in_euler=config.policy_in_euler,
                eval_in_euler=config.eval_in_euler
            )
        else:
            return cls.from_quat(
                prop_dim=config.prop_dim,
                quat_start_indices=config.rotation_start_indices,
                dataset_in_euler=config.from_euler,
                policy_in_euler=config.policy_in_euler,
                eval_in_euler=config.eval_in_euler
            )
        
    @classmethod
    def from_euler(cls, prop_dim: int, euler_start_indices: list[int], dataset_in_euler: bool = True, policy_in_euler: bool = False, eval_in_euler: bool = True):
        """
        Initialize when proprioception uses Euler angles.
        
        Args:
            prop_dim: Total proprioception dimension with Euler (e.g., 14 for two rotations)
            euler_start_indices: Starting indices of Euler angles (e.g., [3, 10])
        """
        num_rotations = len(euler_start_indices)
        
        # Calculate quat indices: each rotation adds 1 to all subsequent indices
        quat_start_indices = []
        for i, euler_idx in enumerate(euler_start_indices):
            # Add i because each previous rotation added 1 dimension
            quat_start_indices.append(euler_idx + i)
        
        return cls(
            euler_dim=prop_dim,
            quat_dim=prop_dim + num_rotations,  # Each 3 euler → 4 quat adds 1 dimension
            euler_start_indices=euler_start_indices,
            quat_start_indices=quat_start_indices,
            dataset_in_euler=dataset_in_euler,
            policy_in_euler=policy_in_euler,
            eval_in_euler=eval_in_euler,
        )
    
    @classmethod
    def from_quat(cls, prop_dim: int, quat_start_indices: list[int], dataset_in_euler: bool = False, policy_in_euler: bool = True, eval_in_euler: bool = False):
        """
        Initialize when proprioception uses quaternions.
        
        Args:
            prop_dim: Total proprioception dimension with quaternion (e.g., 16 for two rotations)
            quat_start_indices: Starting indices of quaternions (e.g., [3, 11])
        """
        num_rotations = len(quat_start_indices)
        
        # Calculate euler indices: each rotation subtracts 1 from all subsequent indices
        euler_start_indices = []
        for i, quat_idx in enumerate(quat_start_indices):
            # Subtract i because each previous rotation removed 1 dimension
            euler_start_indices.append(quat_idx - i)
        
        return cls(
            euler_dim=prop_dim - num_rotations,  # Each 4 quat → 3 euler removes 1 dimension
            quat_dim=prop_dim,
            euler_start_indices=euler_start_indices,
            quat_start_indices=quat_start_indices,
            dataset_in_euler=dataset_in_euler,
            policy_in_euler=policy_in_euler,
            eval_in_euler=eval_in_euler,
        )
    
    def __init__(self, euler_dim: int, quat_dim: int, euler_start_indices: list[int], quat_start_indices: list[int], dataset_in_euler: bool, policy_in_euler: bool, eval_in_euler: bool):
        self.euler_dim = euler_dim
        self.quat_dim = quat_dim
        self.euler_start_indices = euler_start_indices
        self.quat_start_indices = quat_start_indices
        self.num_rotations = len(euler_start_indices)

        self.dataset_in_euler = dataset_in_euler
        self.policy_in_euler = policy_in_euler
        self.eval_in_euler = eval_in_euler
    
    def to_quat(self, prop, is_delta=False):
        """Convert proprioception to quaternion representation."""
        # Already quaternion
        if len(prop) == self.quat_dim:
            return prop
        
        # Convert from Euler
        assert len(prop) == self.euler_dim, f"Expected prop dim {self.euler_dim}, got {len(prop)}"
        
        segments = []
        prev_end = 0
        
        for euler_idx in self.euler_start_indices:
            # Add segment before this rotation
            segments.append(prop[prev_end:euler_idx])
            
            # Convert Euler to quaternion
            euler = prop[euler_idx:euler_idx + 3]

            # Handle absolute and delta euler coordinates differently
            if is_delta:
                # For delta: convert to axis-angle, then to quaternion
                quat = R.from_rotvec(euler).as_quat()  # rotvec is axis*angle
            else:
                # Absolute orientation
                quat = R.from_euler('xyz', euler).as_quat()    
            
            # Add converted quaternion segment
            segments.append(quat)
            
            # Update for next iteration
            prev_end = euler_idx + 3
        
        # Add remaining segment after last rotation
        segments.append(prop[prev_end:])
        
        return np.concatenate([s for s in segments if len(s) > 0])
    
    def to_euler(self, prop, is_delta=False):
        """Convert proprioception to Euler representation."""
        # Already Euler
        if len(prop) == self.euler_dim:
            return prop
        
        # Convert from quaternion
        assert len(prop) == self.quat_dim, f"Expected prop dim {self.quat_dim}, got {len(prop)}"
        
        segments = []
        prev_end = 0
        
        for quat_idx in self.quat_start_indices:
            # Add segment before this rotation
            segments.append(prop[prev_end:quat_idx])
            
            # Convert quaternion to Euler
            quat = prop[quat_idx:quat_idx + 4]

            # Handle euler transformation differently for absolutes and deltas
            if is_delta:
                # For delta: convert to rotation vector (axis-angle)
                # This gives the angular displacement, which approximates delta Euler for small angles
                euler = R.from_quat(quat).as_rotvec()
            else: 
                euler = R.from_quat(quat).as_euler('xyz')  # [roll, pitch, yaw]
            
            # Add converted euler segment
            segments.append(euler)
            
            # Update for next iteration
            prev_end = quat_idx + 4
        
        # Add remaining segment after last rotation
        segments.append(prop[prev_end:])
        
        return np.concatenate([s for s in segments if len(s) > 0])
    
    # Prepares proprioception to be inputted for policy
    def process_for_policy(self, prop, is_delta: bool = None): 
        return self.to_euler(prop, is_delta) if self.policy_in_euler else self.to_quat(prop, is_delta)
    
    # Prepares actions to be inputted for evaluation (and be compatible with env)
    def process_for_eval(self, prop, is_delta: bool = None):
        return self.to_euler(prop, is_delta) if self.eval_in_euler else self.to_quat(prop, is_delta)
