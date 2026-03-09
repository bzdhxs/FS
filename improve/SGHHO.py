"""
SG-HHO: Spectral Group-based Harris Hawks Optimization

This module implements the SG-HHO algorithm for hyperspectral feature selection.

Core innovations:
1. Fixed-window spectral group partitioning
2. Group-level cooperative HHO update
3. Sigmoid-based binary mapping
4. Group-level decoding mechanism

Reference:
    Spectral Group-based Harris Hawks Optimization for Hyperspectral Feature Selection
    in UAV-based Soil Salinity Mapping
"""

import numpy as np
import math
from mealpy.swarm_based.HHO import OriginalHHO


class SpectralGroupHHO(OriginalHHO):
    """
    Spectral Group-based Harris Hawks Optimization.
    
    This class extends the original HHO algorithm to work at the spectral group level
    rather than individual band level, reducing search space dimensionality and
    preserving spectral continuity.
    
    Design Philosophy:
    ------------------
    - Search space: Band-level (n_bands dimensions) for framework compatibility
    - Update mechanism: Group-level cooperative search for structural optimization
    - Decoding: Group selection → Band selection (all bands in selected groups)
    
    This hybrid approach balances standard optimization framework requirements with
    spectral structure preservation, enabling efficient hyperspectral feature selection
    while maintaining spectral continuity.
    
    Parameters
    ----------
    epoch : int, default=200
        Maximum number of iterations
    pop_size : int, default=50
        Population size
    window_size : int, default=8
        Number of consecutive bands in each spectral group
    sigmoid_lambda : float, default=10
        Steepness parameter for sigmoid transfer function
    binary_threshold : float, default=0.5
        Threshold for binarization
    
    Attributes
    ----------
    groups : list
        List of spectral group dictionaries containing start, end, and indices
    n_groups : int
        Number of spectral groups
    n_bands : int
        Total number of spectral bands
    """
    
    def __init__(self, epoch=200, pop_size=50, window_size=8, 
                 sigmoid_lambda=10, binary_threshold=0.5, 
                 start_wl=400, spectral_res=4, **kwargs):
        """Initialize SG-HHO optimizer."""
        super().__init__(epoch=epoch, pop_size=pop_size, **kwargs)
        
        self.window_size = window_size
        self.sigmoid_lambda = sigmoid_lambda
        self.binary_threshold = binary_threshold
        self.start_wl = start_wl
        self.spectral_res = spectral_res
        
        # These will be set during problem setup
        self.groups = None
        self.n_groups = None
        self.n_bands = None
        self.fitness_cache = {}
        
        # Pre-compute Levy flight sigma
        beta = 1.5
        self.levy_sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                          (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    
    def create_spectral_groups(self, n_bands):
        """
        Create fixed-window spectral groups.
        
        Divides n_bands into groups of size window_size. The last group may be
        smaller if n_bands is not divisible by window_size.
        
        Parameters
        ----------
        n_bands : int
            Total number of spectral bands
            
        Returns
        -------
        list
            List of group dictionaries with keys: 'start', 'end', 'indices'
        """
        n_groups = (n_bands + self.window_size - 1) // self.window_size
        groups = []
        
        for i in range(n_groups):
            start = i * self.window_size
            end = min((i + 1) * self.window_size, n_bands)
            groups.append({
                'start': start,
                'end': end,
                'indices': list(range(start, end)),
                'size': end - start
            })
        
        return groups
    
    def initialization(self):
        """
        Initialize population at group level.
        
        Overrides the parent initialization to work with group-level encoding.
        Each individual is a vector of length n_groups (not n_bands).
        """
        if self.pop is None:
            self.pop = []
        
        # Set up groups based on problem dimensionality
        self.n_bands = self.problem.n_dims
        self.groups = self.create_spectral_groups(self.n_bands)
        self.n_groups = len(self.groups)
        
        # Initialize population at group level
        for i in range(self.pop_size):
            # Random initialization in [0, 1] for each group
            group_position = np.random.rand(self.n_groups)
            
            # Optional: Apply VIS-NIR prior (30% of population)
            if i < int(self.pop_size * 0.3):
                group_position = self._apply_vis_nir_prior(group_position)
            
            # Decode to band level for fitness evaluation
            band_position = self._decode_to_bands(group_position)
            
            # Create agent with band-level position
            agent = self.generate_agent(band_position)
            
            # Store group-level position in agent for tracking
            agent.group_position = group_position
            
            self.pop.append(agent)
    
    def _apply_vis_nir_prior(self, group_position):
        """
        Apply VIS-NIR prior knowledge to group initialization.
        
        Increases initial probability for groups in salt-sensitive regions:
        - Blue-Green (400-580nm): moderate sensitivity
        - Red (580-750nm): high sensitivity
        - NIR (750-1000nm): moderate sensitivity
        
        Parameters
        ----------
        group_position : ndarray
            Group-level position vector
            
        Returns
        -------
        ndarray
            Modified group position with prior applied
        """
        for g_idx, group in enumerate(self.groups):
            # Calculate approximate center wavelength using parameterized values
            center_band = (group['start'] + group['end']) / 2
            center_wl = self.start_wl + center_band * self.spectral_res
            
            # Apply prior weights
            if 550 <= center_wl <= 750:  # Red region (high sensitivity)
                group_position[g_idx] *= 1.3
            elif 400 <= center_wl <= 550:  # Blue-Green (moderate)
                group_position[g_idx] *= 1.2
            elif 750 <= center_wl <= 900:  # NIR (moderate)
                group_position[g_idx] *= 1.1
        
        # Clip to [0, 1]
        return np.clip(group_position, 0, 1)
    
    def _sigmoid_transfer(self, group_vector):
        """
        Apply sigmoid transfer function to group vector with center bias.
        
        T(x) = 1 / (1 + exp(-lambda * (x - 0.5)))
        
        This centers the sigmoid at 0.5, so:
        - x < 0.5 → probability < 0.5 (tend to 0)
        - x > 0.5 → probability > 0.5 (tend to 1)
        
        Parameters
        ----------
        group_vector : ndarray
            Continuous group-level position
            
        Returns
        -------
        ndarray
            Sigmoid-transformed values in [0, 1]
        """
        # Center at 0.5 and apply sigmoid
        x = self.sigmoid_lambda * (group_vector - 0.5)
        # Clip for numerical stability
        x = np.clip(x, -50, 50)
        return 1.0 / (1.0 + np.exp(-x))
    
    def _binarize_groups(self, group_vector):
        """
        Binarize group vector using sigmoid transfer and threshold.
        
        Parameters
        ----------
        group_vector : ndarray
            Continuous group-level position
            
        Returns
        -------
        ndarray
            Binary group selection (0 or 1 for each group)
        """
        sigmoid_values = self._sigmoid_transfer(group_vector)
        return (sigmoid_values > self.binary_threshold).astype(int)
    
    def _decode_to_bands(self, group_vector):
        """
        Decode group-level vector to band-level binary vector.
        
        If a group is selected (after binarization), all bands in that group
        are selected.
        
        Parameters
        ----------
        group_vector : ndarray
            Continuous group-level position
            
        Returns
        -------
        ndarray
            Band-level binary vector (0 or 1 for each band)
        """
        binary_groups = self._binarize_groups(group_vector)
        band_vector = np.zeros(self.n_bands)
        
        for g_idx, group in enumerate(self.groups):
            if binary_groups[g_idx] == 1:
                # Select all bands in this group
                band_vector[group['indices']] = 1
        
        return band_vector
    
    def _encode_to_groups(self, band_vector):
        """
        Encode band-level vector to group-level (for reverse operation).
        
        Group value = proportion of selected bands in the group.
        
        Parameters
        ----------
        band_vector : ndarray
            Band-level binary or continuous vector
            
        Returns
        -------
        ndarray
            Group-level continuous vector
        """
        group_vector = np.zeros(self.n_groups)
        
        for g_idx, group in enumerate(self.groups):
            # Calculate proportion of selected bands in this group
            group_bands = band_vector[group['indices']]
            group_vector[g_idx] = np.mean(group_bands)
        
        return group_vector
    
    def evolve(self, epoch):
        """
        Evolve population using group-level HHO update.
        
        This method overrides the parent evolve() to implement group-level
        cooperative search while preserving the original HHO exploration/
        exploitation mechanism.
        
        Parameters
        ----------
        epoch : int
            Current iteration number
        """
        # Update each hawk
        pop_new = []
        for idx in range(self.pop_size):
            # Calculate individual energy parameter E for diversity
            E0 = 2 * np.random.rand() - 1
            E = 2 * E0 * (1 - epoch / self.epoch)
            
            # Get current group-level position
            if hasattr(self.pop[idx], 'group_position'):
                current_group_pos = self.pop[idx].group_position
            else:
                # Fallback: encode from band position
                current_group_pos = self._encode_to_groups(self.pop[idx].solution)
            
            # Get rabbit (best solution) group position
            if hasattr(self.g_best, 'group_position'):
                rabbit_group_pos = self.g_best.group_position
            else:
                rabbit_group_pos = self._encode_to_groups(self.g_best.solution)
            
            # Apply HHO update at group level
            if abs(E) >= 1:
                # Exploration phase
                new_group_pos = self._exploration_phase(current_group_pos, rabbit_group_pos, E)
            else:
                # Exploitation phase
                new_group_pos = self._exploitation_phase(current_group_pos, rabbit_group_pos, E)
            
            # Decode to band level
            new_band_pos = self._decode_to_bands(new_group_pos)
            
            # Create new agent
            new_agent = self.generate_agent(new_band_pos)
            new_agent.group_position = new_group_pos
            
            # Greedy selection
            if self.compare_target(new_agent.target, self.pop[idx].target, self.problem.minmax):
                pop_new.append(new_agent)
            else:
                pop_new.append(self.pop[idx])
        
        # Update population
        self.pop = pop_new
        
        # Update global best and ensure group_position is synchronized
        self.g_best = self.get_best_agent(self.pop)
        if not hasattr(self.g_best, 'group_position'):
            self.g_best.group_position = self._encode_to_groups(self.g_best.solution)
    
    def _exploration_phase(self, current_pos, rabbit_pos, E):
        """
        HHO exploration phase at group level.
        
        Parameters
        ----------
        current_pos : ndarray
            Current group-level position
        rabbit_pos : ndarray
            Best solution group-level position
        E : float
            Energy parameter
            
        Returns
        -------
        ndarray
            New group-level position
        """
        q = np.random.rand()
        r1, r2 = np.random.rand(2)
        
        if q >= 0.5:
            # Random hawk position
            rand_idx = np.random.randint(0, self.pop_size)
            if hasattr(self.pop[rand_idx], 'group_position'):
                rand_pos = self.pop[rand_idx].group_position
            else:
                rand_pos = self._encode_to_groups(self.pop[rand_idx].solution)
            
            new_pos = rand_pos - r1 * np.abs(rand_pos - 2 * r2 * current_pos)
        else:
            # Based on rabbit and random position (simplified)
            r3, r4 = np.random.rand(2)
            new_pos = (rabbit_pos - current_pos) - r3 * r4
        
        # Clip to [0, 1]
        return np.clip(new_pos, 0, 1)
    
    def _exploitation_phase(self, current_pos, rabbit_pos, E):
        """
        HHO exploitation phase at group level.
        
        Parameters
        ----------
        current_pos : ndarray
            Current group-level position
        rabbit_pos : ndarray
            Best solution group-level position
        E : float
            Energy parameter
            
        Returns
        -------
        ndarray
            New group-level position
        """
        r = np.random.rand()
        
        if r >= 0.5 and abs(E) >= 0.5:
            # Soft besiege
            delta_E = rabbit_pos - E * np.abs(rabbit_pos - current_pos)
            new_pos = delta_E - E * np.abs(delta_E - current_pos)
        
        elif r >= 0.5 and abs(E) < 0.5:
            # Hard besiege
            new_pos = rabbit_pos - E * np.abs(rabbit_pos - current_pos)
        
        elif r < 0.5 and abs(E) >= 0.5:
            # Soft besiege with progressive rapid dives
            delta_E = rabbit_pos - E * np.abs(rabbit_pos - current_pos)
            
            # Random jump
            S = np.random.rand(self.n_groups)
            LF = self._levy_flight(self.n_groups)
            Y = rabbit_pos - E * np.abs(rabbit_pos - current_pos)
            Z = Y + S * LF
            
            new_pos = delta_E if np.random.rand() < 0.5 else Z
        
        else:
            # Hard besiege with progressive rapid dives
            S = np.random.rand(self.n_groups)
            LF = self._levy_flight(self.n_groups)
            Y = rabbit_pos - E * np.abs(rabbit_pos - current_pos)
            Z = Y + S * LF
            
            new_pos = Z
        
        # Clip to [0, 1]
        return np.clip(new_pos, 0, 1)
    
    def _levy_flight(self, dim):
        """
        Generate Levy flight step using pre-computed sigma.
        
        Parameters
        ----------
        dim : int
            Dimensionality
            
        Returns
        -------
        ndarray
            Levy flight step vector
        """
        beta = 1.5
        
        # Use pre-computed sigma for efficiency
        u = np.random.normal(0, self.levy_sigma, dim)
        v = np.random.normal(0, 1, dim)
        
        step = u / (np.abs(v) ** (1 / beta))
        return 0.01 * step
