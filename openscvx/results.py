from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
import numpy as np

from openscvx.backend.state import State
from openscvx.backend.control import Control


@dataclass
class OptimizationResults:
    """
    Class to hold optimization results from the SCP solver.
    
    This class replaces the dictionary-based results structure with a more
    structured and type-safe approach.
    """
    
    # Core optimization results
    converged: bool
    t_final: float
    u: Control
    x: State
    
    # History of SCP iterations
    x_history: List[np.ndarray] = field(default_factory=list)
    u_history: List[np.ndarray] = field(default_factory=list)
    discretization_history: List[np.ndarray] = field(default_factory=list)
    J_tr_history: List[np.ndarray] = field(default_factory=list)
    J_vb_history: List[np.ndarray] = field(default_factory=list)
    J_vc_history: List[np.ndarray] = field(default_factory=list)
    
    # Post-processing results (added by propagate_trajectory_results)
    t_full: Optional[np.ndarray] = None
    x_full: Optional[np.ndarray] = None
    u_full: Optional[np.ndarray] = None
    cost: Optional[float] = None
    ctcs_violation: Optional[np.ndarray] = None
    
    # Additional plotting/application data (added by user)
    plotting_data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize the results object."""
        pass
    
    def update_plotting_data(self, **kwargs):
        """
        Update the plotting data with additional information.
        
        Args:
            **kwargs: Key-value pairs to add to plotting_data
        """
        self.plotting_data.update(kwargs)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the results, similar to dict.get().
        
        Args:
            key: The key to look up
            default: Default value if key is not found
            
        Returns:
            The value associated with the key, or default if not found
        """
        # Check if it's a direct attribute
        if hasattr(self, key):
            return getattr(self, key)
        
        # Check if it's in plotting_data
        if key in self.plotting_data:
            return self.plotting_data[key]
        
        return default
    
    def __getitem__(self, key: str) -> Any:
        """
        Allow dictionary-style access to results.
        
        Args:
            key: The key to look up
            
        Returns:
            The value associated with the key
            
        Raises:
            KeyError: If key is not found
        """
        # Check if it's a direct attribute
        if hasattr(self, key):
            return getattr(self, key)
        
        # Check if it's in plotting_data
        if key in self.plotting_data:
            return self.plotting_data[key]
        
        raise KeyError(f"Key '{key}' not found in results")
    
    def __setitem__(self, key: str, value: Any):
        """
        Allow dictionary-style assignment to results.
        
        Args:
            key: The key to set
            value: The value to assign
        """
        # Check if it's a direct attribute
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            # Store in plotting_data
            self.plotting_data[key] = value
    
    def __contains__(self, key: str) -> bool:
        """
        Check if a key exists in the results.
        
        Args:
            key: The key to check
            
        Returns:
            True if key exists, False otherwise
        """
        return hasattr(self, key) or key in self.plotting_data
    
    def update(self, other: Dict[str, Any]):
        """
        Update the results with additional data from a dictionary.
        
        Args:
            other: Dictionary containing additional data
        """
        for key, value in other.items():
            self[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the results to a dictionary for backward compatibility.
        
        Returns:
            Dictionary representation of the results
        """
        result_dict = {}
        
        # Add all direct attributes
        for attr_name in self.__dataclass_fields__:
            if attr_name != 'plotting_data':
                result_dict[attr_name] = getattr(self, attr_name)
        
        # Add plotting data
        result_dict.update(self.plotting_data)
        
        return result_dict 