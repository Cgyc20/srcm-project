import numpy as np
from .reaction import Reaction
from tqdm import tqdm
import json
import os
import contextlib
import io
from joblib import Parallel, delayed


class SSA:
    """
    Stochastic Simulation Algorithm (Gillespie) for a Reaction system.
    
    Tensor Ordering Convention
    ---------------------------
    All tensors in this class follow the convention:
        tensor[time_index, species_index, compartment_index]
    
    This means:
    - Axis 0: Time points (length = number of time steps)
    - Axis 1: Species (length = n_species)
    - Axis 2: Spatial compartments (length = n_compartments)
    
    Example:
        To access species 2 in compartment 5 at time step 10:
            tensor[10, 2, 5]
        
        To get all time points for species 0 in compartment 3:
            tensor[:, 0, 3]
        
        To get spatial distribution of species 1 at time step 20:
            tensor[20, 1, :]
    """

    def __init__(self, reaction_system: Reaction):
        """
        Initialize the SSA simulator.
        
        Parameters
        ----------
        reaction_system : Reaction
            An instance of the Reaction class containing reactions and stoichiometry.
        """
        self.reaction_system = reaction_system
        self.species_list = reaction_system.species_list
        self.n_species = len(self.species_list)
    
        self.species_index = reaction_system.species_index
        self.stoichiometric_matrix = reaction_system.stoichiometric_matrix
        self.number_of_reactions = reaction_system.number_of_reactions
        self.reaction_set = reaction_system.reaction_set
      
    

    def set_conditions(self, 
                   n_compartments: int,
                   domain_length: float,
                   total_time: float,
                   initial_conditions: np.ndarray,
                   timestep: float,
                   Macroscopic_diffusion_rates: list,
                   boundary_conditions: str):
        """
        Set initial conditions and validate inputs.
        
        Parameters
        ----------
        n_compartments : int
            Number of spatial compartments in the domain.
        domain_length : float
            Total length of the spatial domain.
        total_time : float
            Total simulation time.
        initial_conditions : np.ndarray
            Initial molecule counts with shape (n_species, n_compartments).
            - Axis 0: Species index
            - Axis 1: Compartment index
            Example: initial_conditions[species_i, compartment_j] gives the count 
                     of species i in compartment j at t=0.
        timestep : float
            Time interval for recording simulation snapshots.
        Macroscopic_diffusion_rates : list
            List of diffusion rates (one per species), length must equal n_species.
        boundary_conditions : str
            Either 'periodic' or 'zero-flux'.
        
        Notes
        -----
        The internal tensor will be created with shape:
            (n_timepoints, n_species, n_compartments)
        where n_timepoints = len(np.arange(0, total_time, timestep))
        """
        
        if not isinstance(n_compartments, int) or n_compartments <= 0:
            raise ValueError("Number of compartments must be positive")
        
        if not isinstance(domain_length, float):
            raise ValueError("Domain length must be a float")
        
        if not isinstance(total_time, float):
            raise ValueError("Total time must be a float")
        
        # Validate initial_conditions type
        if not isinstance(initial_conditions, np.ndarray):
            raise ValueError("Initial conditions must be a numpy array")

        if initial_conditions.shape != (self.n_species, n_compartments):
            raise ValueError(
                f"Initial conditions must have shape (n_species, n_compartments) = "
                f"({self.n_species}, {n_compartments}), but got {initial_conditions.shape}"
            )

        # Ensure initial_conditions are non-negative integers
        if not np.issubdtype(initial_conditions.dtype, np.integer):
            if np.all(initial_conditions >= 0):
                initial_conditions = initial_conditions.astype(int)
            else:
                raise ValueError("Initial conditions must be non-negative integers")

        self.initial_conditions = initial_conditions
        
      
        if not isinstance(timestep, float):
            raise ValueError("Timestep must be a float")
        
        if not isinstance(Macroscopic_diffusion_rates, list):
            raise ValueError("Diffusion rates must be a list")
        
        if len(Macroscopic_diffusion_rates) != self.n_species:
            raise ValueError("Diffusion rates list length must match number of species")
        
        for rate in Macroscopic_diffusion_rates:
            if not isinstance(rate, float):
                raise ValueError("Each diffusion rate must be a float")
        

        possible_boundary_strings = ['periodic', 'zero-flux']

        if not isinstance(boundary_conditions, str):
            raise ValueError("Boundary conditions must be a string")
        if boundary_conditions not in possible_boundary_strings:
            raise ValueError(f"Boundary conditions must be one of {possible_boundary_strings}")
        

        # If all checks pass, store values
        self.n_compartments = n_compartments
        self.domain_length = domain_length
        self.total_time = total_time
        self.initial_conditions = initial_conditions
        self.timestep = timestep
        self.Macroscopic_diffusion_rates = Macroscopic_diffusion_rates
        self.timevector = np.arange(0, self.total_time, self.timestep)
        self.h = self.domain_length / self.n_compartments
        self.space = np.linspace(0, self.domain_length - self.h, self.n_compartments)
        self.boundary_conditions = boundary_conditions
        self.propensity_vector = np.zeros(
            self.n_compartments * self.n_species + 
            self.n_compartments * self.reaction_system.number_of_reactions
        )
        print("All initial conditions are valid.")

        initial_tensor = self._generate_dataframes()

        self.jump_rate_list = [
            macroscopic_rate / (self.h ** 2) 
            for macroscopic_rate in self.Macroscopic_diffusion_rates
        ]


    def _generate_dataframes(self):
        """
        Generate the result tensor for storing simulation data.
        
        Returns
        -------
        tensor : np.ndarray
            3D array with shape (n_timepoints, n_species, n_compartments).
            
            Dimension ordering:
            - Axis 0 (rows): Time points, length = len(self.timevector)
            - Axis 1 (depth): Species, length = self.n_species
            - Axis 2 (cols): Spatial compartments, length = self.n_compartments
            
            Access pattern:
                tensor[time_idx, species_idx, compartment_idx]
            
            Example usage:
                # Get concentration of species 0 across all compartments at time 5
                spatial_profile = tensor[5, 0, :]
                
                # Get time series of species 1 in compartment 10
                time_series = tensor[:, 1, 10]
                
                # Get all species counts in compartment 0 at time 0
                initial_state_comp0 = tensor[0, :, 0]
        
        Notes
        -----
        The tensor is initialized with zeros and dtype=int. The first time point 
        (tensor[0, :, :]) is filled with self.initial_conditions.
        """

        tensor = np.zeros(
            (len(self.timevector), self.n_species, self.n_compartments), 
            dtype=int
        )

        tensor[0, :, :] = self.initial_conditions
        self.tensor = tensor
        return tensor
    
    def _propensity_calculation(self,
                                dataframe: np.ndarray,
                                propensity_vector: np.ndarray):
        """
        Calculate propensity functions for all reactions and diffusion events.
        
        Parameters
        ----------
        dataframe : np.ndarray
            Current state with shape (n_species, n_compartments).
            - Axis 0: Species index
            - Axis 1: Compartment index
            Access: dataframe[species_idx, compartment_idx]
        
        propensity_vector : np.ndarray
            1D array to store all propensities, with shape:
            (n_compartments * n_species + n_compartments * number_of_reactions,)
            
            Structure:
            - Indices [0 : n_compartments*n_species] : Diffusion propensities
              Ordered as: [species_0_comp_0, species_0_comp_1, ..., 
                          species_1_comp_0, species_1_comp_1, ...]
            - Indices [n_compartments*n_species : end] : Reaction propensities
              Ordered as: [reaction_0_comp_0, reaction_0_comp_1, ...,
                          reaction_1_comp_0, reaction_1_comp_1, ...]
        
        Returns
        -------
        propensity_vector : np.ndarray
            Updated propensity vector with calculated values.
        
        Notes
        -----
        The propensity for diffusion of species i from compartment j is stored at:
            propensity_vector[i * n_compartments + j]
        
        The propensity for reaction r in compartment j is stored at:
            propensity_vector[n_species * n_compartments + r * n_compartments + j]
        """

        assert dataframe.shape == (self.n_species, self.n_compartments), \
            f"Dataframe shape must be (n_species, n_compartments) = " \
            f"({self.n_species}, {self.n_compartments}), got {dataframe.shape}"
        
        assert propensity_vector.shape == (
            self.n_compartments * self.n_species + 
            self.n_compartments * self.reaction_system.number_of_reactions,
        ), "Propensity vector shape is incorrect"
        
        # Calculate diffusion propensities
        for species_index in range(self.n_species):
            corresponding_jump_rate = self.jump_rate_list[species_index]
            start_idx = species_index * self.n_compartments
            end_idx = (species_index + 1) * self.n_compartments
            propensity_vector[start_idx:end_idx] = \
                corresponding_jump_rate * dataframe[species_index, :] * 2.0

        # Calculate reaction propensities
        for i, reaction in enumerate(self.reaction_set):
            start = self.n_compartments * self.n_species + i * self.n_compartments
            end = start + self.n_compartments

            reaction_type = reaction['reaction_type']
            reactant_indices = reaction['reactant_indices']
            rate = reaction['reaction_rate']

            if reaction_type == 'zero_order':
                # Constant production
                propensity_vector[start:end] = rate * self.h

            elif reaction_type == 'first_order':
                idx = reactant_indices[0]
                propensity_vector[start:end] = rate * dataframe[idx, :]

            elif reaction_type == 'second_order':
                if len(reactant_indices) == 1:
                    # Homodimerization: A + A -> products
                    idx = reactant_indices[0]
                    propensity_vector[start:end] = \
                        rate * dataframe[idx, :] * (dataframe[idx, :] - 1) / self.h
                else:
                    # Heterodimerization: A + B -> products
                    idx1, idx2 = reactant_indices
                    propensity_vector[start:end] = \
                        rate * dataframe[idx1, :] * dataframe[idx2, :] / self.h
            else:
                raise ValueError(f"Unknown reaction type {reaction_type}")
            
        return propensity_vector
    

    def _SSA_loop(self):
        """
        Execute the main Gillespie SSA loop.
        
        Returns
        -------
        tensor : np.ndarray
            Simulation results with shape (n_timepoints, n_species, n_compartments).
            
            Dimension ordering:
            - Axis 0: Time (indexed by time step)
            - Axis 1: Species (indexed by species number)
            - Axis 2: Space (indexed by compartment number)
            
            Access: tensor[time_idx, species_idx, compartment_idx]
        
        Notes
        -----
        The algorithm:
        1. Calculates propensities for all possible events
        2. Randomly selects time until next event (tau)
        3. Randomly selects which event occurs
        4. Updates system state
        5. Records state at regular timestep intervals
        """
        
        t = 0.0
        old_time = t

        current_frame = self.initial_conditions.copy()

        while t < self.total_time:

            propensity_vector = self._propensity_calculation(
                dataframe=current_frame,
                propensity_vector=self.propensity_vector
            )

            alpha0 = np.sum(propensity_vector)
            if alpha0 == 0:
                # No further action possible
                break

            r1, r2, r3 = np.random.rand(3)
            tau = (1 / alpha0) * np.log(1 / r1)
            alpha_cum = np.cumsum(propensity_vector)
            index = np.searchsorted(alpha_cum, r2 * alpha0)
            compartment_index = index % self.n_compartments

            # Execute diffusion or reaction based on selected index
            if index < self.n_species * self.n_compartments:
                # Diffusion event
                species_index = index // self.n_compartments
                
                if self.boundary_conditions == 'periodic':
                    if r3 < 0.5:
                        # Move left
                        current_frame[species_index, compartment_index] -= 1
                        current_frame[species_index, 
                                    (compartment_index - 1) % self.n_compartments] += 1
                    else:
                        # Move right
                        current_frame[species_index, compartment_index] -= 1
                        current_frame[species_index, 
                                    (compartment_index + 1) % self.n_compartments] += 1
                
                elif self.boundary_conditions == 'zero-flux':
                    if r3 < 0.5:
                        # Move left
                        if compartment_index == 0:
                            # Reflective boundary - move right instead
                            current_frame[species_index, compartment_index] -= 1 
                            current_frame[species_index, compartment_index + 1] += 1
                        elif compartment_index == self.n_compartments - 1:
                            # Reflective boundary - move left instead
                            current_frame[species_index, compartment_index] -= 1 
                            current_frame[species_index, compartment_index - 1] += 1
                        else:
                            current_frame[species_index, compartment_index] -= 1
                            current_frame[species_index, compartment_index - 1] += 1
                    else:
                        # Move right
                        if compartment_index == self.n_compartments - 1 or compartment_index == 0:
                            # Reflective boundary - no movement
                            pass
                        else:
                            current_frame[species_index, compartment_index] -= 1
                            current_frame[species_index, compartment_index + 1] += 1

            else:
                # Reaction event
                reaction_index = (index - self.n_species * self.n_compartments) // self.n_compartments
                stoichiometric_update = self.stoichiometric_matrix[:, reaction_index]
                current_frame[:, compartment_index] += stoichiometric_update
            
            old_time = t
            t += tau

            # Record state at regular intervals
            ind_before = int(old_time / self.timestep)
            ind_after = int(t / self.timestep)

            if ind_after > ind_before:
                for time_index in range(ind_before + 1, min(ind_after + 1, len(self.timevector))):
                    self.tensor[time_index, :, :] = current_frame

        return self.tensor
    


    def _resolve_n_jobs(self, n_jobs: int, max_n_jobs: int | None) -> int:
        """
        Resolve requested n_jobs against cpu_count and optional max_n_jobs.
        joblib convention: n_jobs=-1 means "all cores".
        """
        cpu = os.cpu_count() or 1

        if n_jobs is None:
            n_jobs_eff = 1
        elif n_jobs == -1:
            n_jobs_eff = cpu
        else:
            n_jobs_eff = int(n_jobs)

        if n_jobs_eff <= 0:
            raise ValueError("n_jobs must be a positive int, or -1 for all cores")

        if max_n_jobs is not None:
            n_jobs_eff = min(n_jobs_eff, int(max_n_jobs))

        return min(n_jobs_eff, cpu)

    def _run_one_repeat(self, seed: int | None = None) -> np.ndarray:
        """
        Worker-safe single repeat: creates a fresh SSA instance (no shared mutable state),
        copies conditions, runs one SSA loop, returns full tensor.
        """
        # Fresh instance to avoid shared mutable state across processes
        sim = SSA(self.reaction_system)

        # Re-apply conditions (validation included). Silence the print inside set_conditions.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            sim.set_conditions(
                n_compartments=self.n_compartments,
                domain_length=self.domain_length,
                total_time=self.total_time,
                initial_conditions=self.initial_conditions.copy(),
                timestep=self.timestep,
                Macroscopic_diffusion_rates=list(self.Macroscopic_diffusion_rates),
                boundary_conditions=self.boundary_conditions,
            )

        if seed is not None:
            np.random.seed(seed)

        sim._generate_dataframes()
        return sim._SSA_loop()

    def _run_one_repeat_final(self, seed: int | None = None) -> np.ndarray:
        """
        Worker-safe single repeat: returns ONLY final frame (n_species, n_compartments).
        """
        tensor = self._run_one_repeat(seed=seed)
        return tensor[-1, :, :].copy()




   
    def run_simulation(
    self,
    n_repeats: int,
    *,
    parallel: bool = False,
    n_jobs: int = -1,
    max_n_jobs: int | None = None,
    progress: bool = True,
    base_seed: int | None = None,
) -> np.ndarray:
        """
        Run multiple SSA simulations and average the results.

        If parallel=True, repeats are distributed across processes using joblib.
        If progress=True, a tqdm progress bar is shown (works in both serial + parallel).
        """
        if n_repeats <= 0:
            raise ValueError("n_repeats must be > 0")

        seeds = None
        if base_seed is not None:
            seeds = [int(base_seed) + i for i in range(n_repeats)]

        # -----------------------
        # Serial (tqdm works normally)
        # -----------------------
        if not parallel:
            summed = np.zeros((len(self.timevector), self.n_species, self.n_compartments), dtype=np.int64)

            iterator = range(n_repeats)
            if progress:
                iterator = tqdm(iterator, total=n_repeats, desc="Running SSA repeats", unit="run", dynamic_ncols=True)

            for i in iterator:
                seed = None if seeds is None else seeds[i]
                self._generate_dataframes()
                if seed is not None:
                    np.random.seed(seed)
                summed += self._SSA_loop()

            return (summed / n_repeats).astype(float)

        # -----------------------
        # Parallel (stream results -> tqdm works)
        # -----------------------
        n_jobs_eff = self._resolve_n_jobs(n_jobs=n_jobs, max_n_jobs=max_n_jobs)
        print(f"[SSA] Running in parallel with {n_jobs_eff} process(es)")

        def one(i: int):
            seed = None if seeds is None else seeds[i]
            tensor = self._run_one_repeat(seed=seed)  # shape (T, S, K)
            return tensor.astype(np.int64, copy=False)

        tasks = (delayed(one)(i) for i in range(n_repeats))

        # stream results as they finish
        results_iter = Parallel(n_jobs=n_jobs_eff, backend="loky", return_as="generator")(tasks)

        if progress:
            try:
                from tqdm.auto import tqdm as tqdm_auto
                results_iter = tqdm_auto(
                    results_iter,
                    total=n_repeats,
                    desc="Running SSA repeats",
                    unit="run",
                    dynamic_ncols=True,
                )
            except Exception:
                pass

        summed = np.zeros((len(self.timevector), self.n_species, self.n_compartments), dtype=np.int64)
        for tensor in results_iter:
            summed += tensor

        return (summed / n_repeats).astype(float)


    def save_simulation_data(self, filename: str, simulation_result: np.ndarray):
        """
        Save SSA simulation data and metadata to a compressed .npz file.

        Parameters
        ----------
        filename : str
            Full path where the file should be saved (must include .npz extension).
        simulation_result : np.ndarray
            Simulation result from run_simulation() with shape 
            (n_timepoints, n_species, n_compartments).
            
            IMPORTANT: Tensor ordering in saved file:
            - Axis 0: Time
            - Axis 1: Species
            - Axis 2: Compartments
            
            When loading the file:
                data = np.load('filename.npz')
                results = data['simulation_result']
                # Access as: results[time_idx, species_idx, compartment_idx]
        
        Saved Arrays
        ------------
        simulation_result : np.ndarray
            Main results (n_timepoints, n_species, n_compartments)
        timevector : np.ndarray
            Time points corresponding to axis 0 of simulation_result
        space : np.ndarray  
            Spatial coordinates corresponding to axis 2 of simulation_result
        domain_length : float
            Total length of spatial domain
        total_time : float
            Total simulation time
        timestep : float
            Time interval between recorded points
        h : float
            Compartment width (domain_length / n_compartments)
        n_species : int
            Number of species (length of axis 1)
        n_compartments : int
            Number of compartments (length of axis 2)
        jump_rates : np.ndarray
            Microscopic jump rates for each species
        reaction_data : str
            JSON string containing reaction information
        
        Example
        -------
        >>> # Save data
        >>> ssa.save_simulation_data('results.npz', results)
        >>> 
        >>> # Load data
        >>> data = np.load('results.npz')
        >>> results = data['simulation_result']  # shape: (time, species, compartments)
        >>> times = data['timevector']
        >>> positions = data['space']
        >>> 
        >>> # Plot species 0 at final time
        >>> plt.plot(positions, results[-1, 0, :])
        """
       
        # Convert reaction set into JSON-serializable format
        reaction_data = []
        for r in self.reaction_set:
            reaction_data.append({
                'reactants': r.get('reactants', {}),
                'products': r.get('products', {}),
                'reaction_type': r.get('reaction_type', ''),
                'reaction_rate': r.get('reaction_rate', 0.0)
            })

        # Save everything into a .npz file
        np.savez_compressed(
            filename,
            simulation_result=simulation_result,
            timevector=self.timevector,
            space=self.space,
            domain_length=self.domain_length,
            total_time=self.total_time,
            timestep=self.timestep,
            h=self.h,
            n_species=self.n_species,
            n_compartments=self.n_compartments,
            jump_rates=np.array(self.jump_rate_list),
            reaction_data=json.dumps(reaction_data)
        )

        print(f"Simulation data successfully saved to {filename}")
        print(f"Tensor shape: {simulation_result.shape} = "
              f"(time={len(self.timevector)}, species={self.n_species}, "
              f"compartments={self.n_compartments})")
        
   
    def run_final_frames(
        self,
        n_repeats: int,
        *,
        progress: bool = True,
        parallel: bool = False,
        n_jobs: int = -1,
        max_n_jobs: int | None = None,
        base_seed: int | None = None,
    ) -> np.ndarray:
        """
        Run multiple SSA simulations and return ONLY the final frame from each repeat.
        """
        if n_repeats <= 0:
            raise ValueError("n_repeats must be > 0")

        seeds = None
        if base_seed is not None:
            seeds = [int(base_seed) + i for i in range(n_repeats)]

        if not parallel:
            final_frames = np.zeros((n_repeats, self.n_species, self.n_compartments), dtype=int)

            iterator = range(n_repeats)
            if progress:
                iterator = tqdm(iterator, desc="Running simulations (final only)")

            for i in iterator:
                seed = None if seeds is None else seeds[i]
                self._generate_dataframes()
                if seed is not None:
                    np.random.seed(seed)
                tensor = self._SSA_loop()
                final_frames[i, :, :] = tensor[-1, :, :]

            return final_frames

        # Parallel branch
        n_jobs_eff = self._resolve_n_jobs(n_jobs=n_jobs, max_n_jobs=max_n_jobs)

        frames = Parallel(n_jobs=n_jobs_eff, backend="loky")(
            delayed(self._run_one_repeat_final)(None if seeds is None else seeds[i])
            for i in range(n_repeats)
        )

        return np.asarray(frames, dtype=int)

        """
        Run multiple SSA simulations and return ONLY the final frame from each repeat.

        Parameters
        ----------
        n_repeats : int
            Number of independent simulation runs.
        progress : bool
            If True, show a tqdm progress bar (requires tqdm).

        Returns
        -------
        final_frames : np.ndarray
            Array of shape (n_repeats, n_species, n_compartments) containing the final
            state from each repeat.

            Dimension ordering:
            - Axis 0: Repeat index
            - Axis 1: Species index
            - Axis 2: Compartment index

            Access pattern:
                final_frames[repeat_idx, species_idx, compartment_idx]
        """
        if n_repeats <= 0:
            raise ValueError("n_repeats must be > 0")

        final_frames = np.zeros((n_repeats, self.n_species, self.n_compartments), dtype=int)

        iterator = range(n_repeats)
        if progress:
            try:
                iterator = tqdm(iterator, desc="Running simulations (final only)")
            except Exception:
                # tqdm not available or misconfigured; silently fall back
                iterator = range(n_repeats)

        for r in iterator:
            self._generate_dataframes()  # Reset tensor for each repeat
            tensor = self._SSA_loop()
            final_frames[r, :, :] = tensor[-1, :, :]

        return final_frames

    def save_final_frames(self, filename: str, final_frames: np.ndarray):
        """
        Save SSA final-frame ensemble data and metadata to a compressed .npz file.

        Parameters
        ----------
        filename : str
            Full path where the file should be saved (must include .npz extension).
        final_frames : np.ndarray
            Output of run_final_frames() with shape (n_repeats, n_species, n_compartments).

        Saved Arrays
        ------------
        final_frames : np.ndarray
            Shape (n_repeats, n_species, n_compartments)
        time_final : float
            Final recorded time (self.timevector[-1])
        space : np.ndarray
            Spatial coordinates (length n_compartments)
        domain_length : float
        total_time : float
        timestep : float
        h : float
        n_species : int
        n_compartments : int
        jump_rates : np.ndarray
        reaction_data : str
            JSON string containing reaction information
        """
        if not isinstance(final_frames, np.ndarray):
            raise ValueError("final_frames must be a numpy array")
        if final_frames.shape[1:] != (self.n_species, self.n_compartments):
            raise ValueError(
                f"final_frames must have shape (n_repeats, {self.n_species}, {self.n_compartments}), "
                f"but got {final_frames.shape}"
            )

        # Convert reaction set into JSON-serializable format
        reaction_data = []
        for r in self.reaction_set:
            reaction_data.append({
                'reactants': r.get('reactants', {}),
                'products': r.get('products', {}),
                'reaction_type': r.get('reaction_type', ''),
                'reaction_rate': r.get('reaction_rate', 0.0)
            })

        np.savez_compressed(
            filename,
            final_frames=final_frames,
            time_final=float(self.timevector[-1]) if len(self.timevector) else float(self.total_time),
            space=self.space,
            domain_length=self.domain_length,
            total_time=self.total_time,
            timestep=self.timestep,
            h=self.h,
            n_species=self.n_species,
            n_compartments=self.n_compartments,
            jump_rates=np.array(self.jump_rate_list),
            reaction_data=json.dumps(reaction_data),
        )

        print(f"Final-frame ensemble successfully saved to {filename}")
        print(f"Final_frames shape: {final_frames.shape} = "
              f"(repeats={final_frames.shape[0]}, species={self.n_species}, "
              f"compartments={self.n_compartments})")

    def run_trajectories(
                        self,
                        n_repeats: int,
                        *,
                        parallel: bool = False,
                        n_jobs: int = -1,
                        max_n_jobs: int | None = None,
                        progress: bool = True,
                        base_seed: int | None = None,
                        dtype=np.int64,
                    ) -> np.ndarray:
        """
        Run multiple SSA simulations and return the FULL trajectory tensor from each repeat.

        Returns
        -------
        trajectories : np.ndarray
            Shape (n_repeats, n_timepoints, n_species, n_compartments)
            Ordering: trajectories[repeat, time, species, compartment]
        """
        if n_repeats <= 0:
            raise ValueError("n_repeats must be > 0")

        seeds = None
        if base_seed is not None:
            seeds = [int(base_seed) + i for i in range(n_repeats)]

        T = len(self.timevector)
        S = self.n_species
        K = self.n_compartments

        # -----------------------
        # Serial
        # -----------------------
        if not parallel:
            trajectories = np.empty((n_repeats, T, S, K), dtype=dtype)

            iterator = range(n_repeats)
            if progress:
                iterator = tqdm(iterator, total=n_repeats, desc="Running SSA trajectories", unit="run", dynamic_ncols=True)

            for i in iterator:
                seed = None if seeds is None else seeds[i]
                self._generate_dataframes()
                if seed is not None:
                    np.random.seed(seed)
                trajectories[i] = self._SSA_loop().astype(dtype, copy=False)

            return trajectories

        # -----------------------
        # Parallel
        # -----------------------
        n_jobs_eff = self._resolve_n_jobs(n_jobs=n_jobs, max_n_jobs=max_n_jobs)
        print(f"[SSA] Running trajectories in parallel with {n_jobs_eff} process(es)")

        def one(i: int):
            seed = None if seeds is None else seeds[i]
            tensor = self._run_one_repeat(seed=seed)  # (T, S, K)
            return i, tensor.astype(dtype, copy=False)

        tasks = (delayed(one)(i) for i in range(n_repeats))
        results_iter = Parallel(n_jobs=n_jobs_eff, backend="loky", return_as="generator")(tasks)

        if progress:
            try:
                from tqdm.auto import tqdm as tqdm_auto
                results_iter = tqdm_auto(
                    results_iter,
                    total=n_repeats,
                    desc="Running SSA trajectories",
                    unit="run",
                    dynamic_ncols=True,
                )
            except Exception:
                pass

        trajectories = np.empty((n_repeats, T, S, K), dtype=dtype)
        for i, tensor in results_iter:
            trajectories[i] = tensor

        return trajectories
