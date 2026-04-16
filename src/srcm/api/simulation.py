from integrated_srcm_ssa import SRCMRunner as _OldRunner

class Simulation:
    def __init__(self, species, boundary="zero-flux"):
        self._runner = _OldRunner(species=species, boundary=boundary)

    def diffusion(self, **kwargs):
        self._runner.define_diffusion(**kwargs)
        return self

    def rates(self, **kwargs):
        self._runner.define_rates(**kwargs)
        return self

    def reaction(self, reactants, products, rate):
        self._runner.add_reaction(reactants, products, rate)
        return self

    def conversion(self, **kwargs):
        self._runner.define_conversion(**kwargs)
        return self

    def pde(self, func):
        self._runner.set_pde_reactions(func)
        return self

    def run_ssa(self, **kwargs):
        return self._runner.run_ssa(**kwargs)

    def run_hybrid(self, **kwargs):
        return self._runner.run_hybrid(**kwargs)