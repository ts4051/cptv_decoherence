'''
Using DeepCore public oscillations data to reprocuce oscillations fit

Based on https://github.com/icecube/pisa/blob/master/pisa_examples/IceCube_3y_oscillations_example.ipynb
'''

import numpy as np
from uncertainties import unumpy as unp
import matplotlib.pyplot as plt
import copy
import pisa
from pisa.core.distribution_maker import DistributionMaker
from pisa.core.pipeline import Pipeline
from pisa.analysis.analysis import Analysis
from pisa import FTYPE, ureg

from deimos.utils.plotting import *

#
# Globals
#

METRIC = "chi2"


#
# Load model (e.g. the MC expectation for our data)
#

# We can now instantiate the `model` (given our configs) that we later fit to data. This now containes two `Pipelines` in a `DistributionMaker`, one for our neutrinos, and one for the background muons.
# template_maker = DistributionMaker(["settings/pipeline/IceCube_3y_neutrinos.cfg", "settings/pipeline/IceCube_3y_muons.cfg"])
template_maker = DistributionMaker(["cptv_decoherence/settings/pipeline_deepcore_nu_mc.cfg", "cptv_decoherence/settings/pipeline_deepcore_mu_mc.cfg"])
print(template_maker)

# The two pipelines are quite different, with most complexity in the neutrino pipeline, that has several `Stage`s and free parameters:
print(template_maker.pipelines[0])

# While the muon pipleine is rather simple
print(template_maker.pipelines[1])


#
# Choose free parameters
#

# The templare aker has a number of model parameters, both physics and nuisance parameters
# Choose which to use here
params_to_use = [
    # Flux
    # 'nue_numu_ratio',
    # 'Barr_uphor_ratio',
    # 'Barr_nu_nubar_ratio',
    'delta_index',
    # Oscillations
    # 'theta13',
    'theta23',
    'deltam31',
    # Norm
    'aeff_scale',
    # 'nu_nc_norm',
    # Atmospheric muon background
    # 'atm_muon_scale',
    # Detector systematics
    # "opt_eff_overall",
    # "opt_eff_lateral",
    # "opt_eff_headon",
    # "ice_scattering",
    # "ice_absorption",
]

# Now fix/free these params in the model
for pipeline in template_maker.pipelines :
    for param in pipeline.params :
        if param.name in params_to_use :
            param.is_fixed = False
        else :
            param.is_fixed = True

print("\nFree params :")
for p in template_maker.params.free :
    print("  %s" % p.name)
print("")


#
# Get analysis histograms
# 

# We can get individual outputs from just a pipleine like so. This fetches outputs from the neutrino pipleine, which are 12 maps.
maps = template_maker.get_outputs()[0] #TODO This is neutrinos, also get muons

# Plot these histograms
fig, axes = plt.subplots(3,4, figsize=(24,10))
plt.subplots_adjust(hspace=0.5)
axes = axes.flatten()
for m, ax in zip(maps, axes):
    m.plot(ax=ax)
fig.tight_layout()

# Also plot the total expectation for the full model (all neutrinos + muons):
template_maker.get_outputs(return_sum=True).plot()


#
# Load real detector data
#

# We can load the real observed data too. This is a Pipeline with no free parameters, as the data is of course fixed.
# NB: When developping a new analysis you will **not** be allowed to look at the data as we do here before the box opening (c.f. *blindness*).

# Load the data pipeline
# data_maker = Pipeline("settings/pipeline/IceCube_3y_data.cfg")
data_maker = Pipeline("cptv_decoherence/settings/pipeline_deepcore_data.cfg")
print(data_maker)

# Run it to get the data histogram
data = data_maker.get_outputs()

# Plot data, and data-MC comparison
fig, ax = plt.subplots(1, 3, figsize=(20, 3))
template_maker.reset_free()
nominal = template_maker.get_outputs(return_sum=True)
data.plot(ax=ax[0], title="Data")
nominal.plot(ax=ax[1], title="Model")
(data - nominal).plot(ax=ax[2], symm=True, title="Diff")
fig.tight_layout()



#
# Fitting/scanning
#
 
# For fitting we need to configure a minimizer, several standard cfgs are available, but you can also define your own.
# For the fit we need to choose a `metric`, and by default, theta23 octants, which are quasi degenerate, are fit seperately, which means two fits are run.

# Define minimizer settings
# minimizer_cfg = pisa.utils.fileio.from_file('settings/minimizer/slsqp_ftol1e-6_eps1e-4_maxiter1000.json')
minimizer_cfg = pisa.utils.fileio.from_file('cptv_decoherence/settings/minimizer.json')

# Define output file
results_file = os.path.abspath("results.json")

# Create analysis class
ana = Analysis()

# Choose whether to scan, or fit
scan = True

if scan :

    #
    # Scan
    #

    result = ana.scan(
        data_dist=data,
        hypo_maker=template_maker,
        metric=METRIC,
        minimizer_settings=minimizer_cfg,
        fit_octants_separately=True, # This fits both theta23 octants separately and chooses the best
        param_names=["theta23", "deltam31"],
        values=[[40., 45., 50.]*ureg["degree"], [2.4e-3, 2.5e-3, 2.6e-3]*ureg["eV**2"]],
        outer=True, # Produce a grid from the scan values
        outfile=results_file,
    )

    print(result)

else :

    #
    # Fit
    #

    result = ana.fit_hypo(
        data_dist=data,
        hypo_maker=template_maker,
        metric=METRIC,
        minimizer_settings=minimizer_cfg,
        fit_octants_separately=True, # This fits both theta23 octants separately and chooses the best
    )

    # Save to file
    pisa.utils.fileio.to_file(result, results_file) # Save

    # Report fit results
    bestfit_params = result[0]['params'].free
    print("\nFit results :")
    for p in bestfit_params :
        print("  %s = %s (nominal = %s)" % (p.name, p.value, p.nominal_value))
    print("  %s = %s" % (result[0]["metric"], result[0]["metric_val"]))
    print("")


#
# Done
#

# Report
print("\nFit results : %s" % results_file)

# Dump plots
print("")
dump_figures_to_pdf( __file__.replace(".py", ".pdf") )
