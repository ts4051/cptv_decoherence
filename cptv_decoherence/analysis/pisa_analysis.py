'''
A helper class for running an analysis using PISA

Tom Stuttard, Christoph Ternes
'''

#
# Analysis class
#

'''
Using IceCube public oscillations data to reprocuce oscillations fit

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

# Define minimizer metric
METRIC = "chi2"

# Choose which params are free
PARAMS_TO_USE = [
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


#
# IceCube analysis class
#

class IceCubeAnalysis(Analysis) :

    def __init__(self, output_dir, real_data=False, osc_solver=None) :

        # Store args
        self.output_dir = output_dir
        self.real_data = real_data
        self.osc_solver = osc_solver

        #
        # Basic config
        #

        # Directory for output files
        self.output_dir = os.path.abspath(self.output_dir)
        assert not os.path.exists(self.output_dir), "Output directory already exists : %s" % self.output_dir
        os.makedirs(self.output_dir)

        # Define where to find minimizer configuration
        # self.minimizer_cfg = pisa.utils.fileio.from_file('cptv_decoherence/settings/minimizer.json')
        self.minimizer_cfg = pisa.utils.fileio.from_file('settings/minimizer/slsqp_ftol1e-6_eps1e-4_maxiter1000.json')
        
        # Other minimizer config
        self.fit_octants_separately = True # This fits both theta23 octants separately and chooses the best


        #
        # Load model (e.g. the MC expectation for our data)
        #

        # Choose osc solver
        if self.osc_solver is None :
            self.osc_solver = "prob3"
        assert self.osc_solver in ["prob3", "nusquids", "deimos"], "Unknown osc solver : %s" % self.osc_solver

        # We can now instantiate the `model` (given our configs) that we later fit to data. This now containes two `Pipelines` in a `DistributionMaker`, one for our neutrinos, and one for the background muons.
        # template_maker = DistributionMaker(["settings/pipeline/IceCube_3y_neutrinos.cfg", "settings/pipeline/IceCube_3y_muons.cfg"])
        self.template_cfgs = ["cptv_decoherence/settings/pipeline_deepcore_nu_mc_%s.cfg"%self.osc_solver, "cptv_decoherence/settings/pipeline_deepcore_mu_mc.cfg"]
        self.template_maker = DistributionMaker(self.template_cfgs)
        print("\nTemplate maker:")
        print(self.template_maker)
        for pipeline in self.template_maker.pipelines :
            print("\n%s" % str(pipeline))


        #
        # Load data
        #

        # If using real data, load the pipeline
        # Otherwise, use the template maker for making pseuodata
        if self.real_data :
            self.data_cfg = "cptv_decoherence/settings/pipeline_deepcore_data.cfg"
            self.data_maker = DistributionMaker(self.data_cfg)
            print("\nData maker:")
            print(self.data_maker)
        else :
            print("Using pseuodata")
            self.data_maker = DistributionMaker(self.template_cfgs) # Use template cfg for pseudodata

        # Now get the actual data from the data maker
        self.data = self.data_maker.get_outputs(return_sum=True)


        #
        # Choose free parameters
        #

        # Now fix/free these params in the model
        for pipeline in self.template_maker.pipelines :
            for param in pipeline.params :
                if param.name in PARAMS_TO_USE :
                    param.is_fixed = False
                else :
                    param.is_fixed = True

        print("\nFree params :")
        for p in self.template_maker.params.free :
            print("  %s" % p.name)
        print("")


    def plot_hists(self) :
        '''
        Plot the data and template histograms
        '''

        #TODO

        # # We can get individual outputs from just a pipleine like so. This fetches outputs from the neutrino pipleine, which are 12 maps.
        # maps = template_maker.get_outputs()[0] #TODO This is neutrinos, also get muons

        # # Plot these histograms
        # fig, axes = plt.subplots(3,4, figsize=(24,10))
        # plt.subplots_adjust(hspace=0.5)
        # axes = axes.flatten()
        # for m, ax in zip(maps, axes):
        #     m.plot(ax=ax)
        # fig.tight_layout()

        # # Also plot the total expectation for the full model (all neutrinos + muons):
        # template_maker.get_outputs(return_sum=True).plot()


        # # Plot data, and data-MC comparison
        # fig, ax = plt.subplots(1, 3, figsize=(20, 3))
        # template_maker.reset_free()
        # nominal = template_maker.get_outputs(return_sum=True)
        # data.plot(ax=ax[0], title="Data")
        # nominal.plot(ax=ax[1], title="Model")
        # (data - nominal).plot(ax=ax[2], symm=True, title="Diff")
        # fig.tight_layout()


    def fit(self) :
        '''
        Fit the model to the data
        '''

        # Define output file
        results_file = os.path.join(self.output_dir, "fit.json")

        # Fit
        result = super(IceCubeAnalysis, self).fit_hypo(
            data_dist=self.data,
            hypo_maker=self.template_maker,
            metric=METRIC,
            minimizer_settings=self.minimizer_cfg,
            fit_octants_separately=self.fit_octants_separately,
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

        return result


    def scan(
        self,
        param_names,
        param_values,
    ) :
        '''
        Scan across the requested params, and fit for each case
        '''

        # Define output file
        results_file = os.path.join(self.output_dir, "scan_{index:06n}.json")

        # Scan
        result = super(IceCubeAnalysis, self).scan(
            data_dist=self.data,
            hypo_maker=self.template_maker,
            metric=METRIC,
            minimizer_settings=self.minimizer_cfg,
            fit_octants_separately=self.fit_octants_separately,
            param_names=param_names,
            values=param_values,
            outer=True, # Produce a grid from the scan values
            outfile=results_file,
        )

        return result

