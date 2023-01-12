class SMLAOptimizer:
	pass	
from .linear_shatter import LinearShatterBFOptimizer
from .cma import CMAESOptimizer
# from .cma_feasible import CMAESFeasibleOptimizer
from .bfgs import BFGSOptimizer
# from .trust_const import TrustConstOptimizer
# from .shgo import SHGOOptimizer
# from .minimize import MinimizeOptimizer

OPTIMIZERS = { opt.cli_key():opt for opt in SMLAOptimizer.__subclasses__() }