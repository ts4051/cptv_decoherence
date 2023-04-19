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


# ## Model

# We can now instantiate the `model` (given our configs) that we later fit to data. This now containes two `Pipelines` in a `DistributionMaker`, one for our neutrinos, and one for the background muons.

# In[24]:


model = DistributionMaker(["settings/pipeline/IceCube_3y_neutrinos.cfg", "settings/pipeline/IceCube_3y_muons.cfg"])
# this just turns on profiling
model.profile = True
model


# Our model has a number of free parameters, that will be used in our fit to the data

# In[25]:


model.params.free


# The two pipelines are quite different, with most complexity in the neutrino pipeline, that has several `Stage`s and free parameters:

# In[26]:


model.pipelines[0]


# In[27]:


model.pipelines[0].stages[2].params


# While the muon pipleine is rather simple

# In[28]:


model.pipelines[1]


# ## Retrieve Outputs
# 
# We can get individual outputs from just a pipleine like so. This fetches outputs from the neutrino pipleine, which are 12 maps.

# In[29]:


maps = model.pipelines[0].get_outputs()


# In[30]:


maps.names


# In[31]:


fig, axes = plt.subplots(3,4, figsize=(24,10))
plt.subplots_adjust(hspace=0.5)
axes = axes.flatten()

for m, ax in zip(maps, axes):
    m.plot(ax=ax)


# If we are interested in just the total expecatation from the full model (all neutrinos + muons), we can do the following:

# In[32]:


model.get_outputs(return_sum=True).plot()


# ## Diff plots
# 
# Let's explore how a change in one of our nuisance parameters affects the expected counts per bin. Here we choose a *hole ice* parameter and move it a smidge.

# In[33]:


# reset all free parameters to put them back to nominal values
model.reset_free()
nominal = model.get_outputs(return_sum=True)

# shift one parameter
model.params.opt_eff_lateral.value = 20
sys = model.get_outputs(return_sum=True)


# In[34]:


((nominal[0] - sys[0])/nominal[0]).plot(symm=True, clabel="rel. difference")


# ## Get Data
# 
# We can load the real observed data too. This is a Pipeline with no free parameters, as the data is of course fixed.
# NB: When developping a new analysis you will **not** be allowed to look at the data as we do here before the box opening (c.f. *blindness*).

# In[35]:


# real data
data_maker = Pipeline("settings/pipeline/IceCube_3y_data.cfg")
data = data_maker.get_outputs()


# In[36]:


data_maker


# In[37]:


fig, ax = plt.subplots(1, 3, figsize=(20, 3))

model.reset_free()
nominal = model.get_outputs(return_sum=True)

data.plot(ax=ax[0], title="Data")
nominal.plot(ax=ax[1], title="Model")
(data - nominal).plot(ax=ax[2], symm=True, title="Diff")


# ## Fitting
# 
# For fitting we need to configure a minimizer, several standard cfgs are available, but you can also define your own.
# For the fit we need to choose a `metric`, and by default, theta23 octants, which are quasi degenerate, are fit seperately, which means two fits are run.

# In[39]:


minimizer_cfg = pisa.utils.fileio.from_file('settings/minimizer/slsqp_ftol1e-6_eps1e-4_maxiter1000.json')
ana = Analysis()


# In[41]:


get_ipython().run_cell_magic('time', '', "result = ana.fit_hypo(\n         data,\n         model,\n         metric='mod_chi2',\n         minimizer_settings=minimizer_cfg,\n         fit_octants_separately=True,\n        )")


# Here we can view the bestfit parameters - the result of our fit.
# We have run two fits (separately for each theta23 octant), and the best result is stored in `results[0]` (both fits are also available under `results[1]`)

# In[18]:


bestfit_params = result[0]['params'].free
bestfit_params


# In[19]:


# update the model with the bestfit (make a copy here, because we don't want our bestfit params to be affected (NB: stuff is passed by reference in python))
model.update_params(copy.deepcopy(bestfit_params))


# Let's see how good that fit looks like. We here construct signed mod_chi2 maps by hand.
# You can see that after the fit, it improved considerably, and the distribution of chi2 values is now more uniform - not much features can be seen anymore.

# In[20]:


fig, ax = plt.subplots(2, 3, figsize=(20, 7))
plt.subplots_adjust(hspace=0.5)

bestfit = model.get_outputs(return_sum=True)

data.plot(ax=ax[0,0], title="Data")
nominal.plot(ax=ax[0,1], title="Nominal")
diff = data - nominal
(abs(diff)*diff/(nominal + unp.std_devs(nominal.hist['total']))).plot(ax=ax[0,2], symm=True, title=r"signed $\chi^2$", vmin=-12, vmax=12)

data.plot(ax=ax[1,0], title="Data")
bestfit.plot(ax=ax[1,1], title="Bestfit")
diff = data - bestfit
(abs(diff)*diff/(bestfit + unp.std_devs(bestfit.hist['total']))).plot(ax=ax[1,2], symm=True, title=r"signed $\chi^2$", vmin=-12, vmax=12)


# When checking the chi2 value from the fitted model, you maybe see that it is around 113, while in the minimizer loop we saw it converged to 116. It is important to keep in mind that in the fit we had extended the metric with prior penalty terms. When we add those back we get the identical number as reported in the fit.

# In[21]:


print(data.metric_total(nominal, 'mod_chi2'))
print(data.metric_total(bestfit, 'mod_chi2'))


# Evaluating other metrics just for fun:

# In[22]:


for metric in pisa.utils.stats.ALL_METRICS:
    try:
        print('%s = %.3f'%(metric,data.metric_total(bestfit, metric)))
    except:
        print('%s failed'%metric)


# Adding prior penalty terms

# In[23]:


model.update_params(copy.deepcopy(bestfit_params))
data.metric_total(bestfit, 'mod_chi2') + model.params.priors_penalty('mod_chi2')


# In[24]:


result[0]['metric_val']


# ## Storing the results to a file
# 
# Since the fit took a while, it might be useful to store the results to a file. (NB: in this example we use a temp file, but in real life you would of course just use a real pathname instead!)

# In[25]:


import tempfile
temp = tempfile.NamedTemporaryFile(suffix='.json')
pisa.utils.fileio.to_file(result, temp.name)


# To reload, we can read the file. But to get PISA objects back, they need to instantiated

# In[26]:


result_reload = pisa.utils.fileio.from_file(temp.name)
bestfit_params = pisa.core.ParamSet(result_reload[0]['params']).free
bestfit_params
temp.close()


# ## Profiling
# 
# To understand what parts of the model were executed how many times, and how long it took, the profiling  can help:

# In[27]:


model.pipelines[0].report_profile()


# In[ ]:




