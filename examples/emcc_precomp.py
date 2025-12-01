import pandas as pd

from bcause.models.cmodel import StructuralCausalModel
from bcause.learning.parameter.expectation_maximization import ExpectationMaximizationPrecomputed
from bcause.util.watch import Watch
from bcause.factors import MultinomialFactor

# Import filepath in models
filepath = "./models/scm/g3_model_45.bif"
datapath = "./models/data/g3_data_45.csv"
model = StructuralCausalModel.read(filepath)
data = pd.read_csv(datapath, dtype='str')

# Set all factors to list type
for var_name, factor in model.factors.items():
    model.factors[var_name] = MultinomialFactor(domain=factor.domain, values=factor.values, left_vars=factor.left_vars,
                                     right_vars=factor.right_vars, vtype="list")

# Run Precomputed EM
Watch.start()
em_precomp= ExpectationMaximizationPrecomputed(model, ignore_convergence=True,as_list=True)
em_precomp.run(data[model.endogenous],max_iter=50)
Watch.stop_print()