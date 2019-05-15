"""
Plotting brain images with Nilearn
==================================

Nilearn (https://nilearn.github.io/) has fetchers to download online
neuroimaging datasets and many tools to create interactive or
publication-quality visualizations.

"""

######################################################################
import warnings
warnings.simplefilter('ignore')

######################################################################
# Statistical maps
# ================

######################################################################
# Download and plot an individual-level statistical map and plot it on the
# subject's T1 image:

from nilearn import datasets, plotting
localizer = datasets.fetch_localizer_button_task()
plotting.view_img(localizer['tmap'], bg_img=localizer['anat'], threshold='97%')


######################################################################
# Download and plot a group-level statistical map:

img = datasets.fetch_neurovault_motor_task()['images'][0]
plotting.view_img(img, threshold='95%')


######################################################################
# More about dataset downloaders: https://nilearn.github.io/modules/reference.html#module-nilearn.datasets
#
# More about plotting: https://nilearn.github.io/plotting/index.html

######################################################################
# Static plots that can be saved in a variety of formats:

plotting.plot_stat_map(img, threshold=3)


######################################################################
plotting.plot_stat_map(
    img, cut_coords=[-18, 64], display_mode='z', threshold=3.)


######################################################################
plotting.plot_glass_brain(img, plot_abs=False, threshold=3.)


######################################################################
# Visualize projections on the cortical surface
# ---------------------------------------------

plotting.view_img_on_surf(img, threshold='95%', surf_mesh='fsaverage')


######################################################################
# Atlases
# =======

destrieux = datasets.fetch_atlas_destrieux_2009()
plotting.view_img(destrieux['maps'],
                  resampling_interpolation='nearest',
                  cmap='gist_ncar', symmetric_cmap=False, colorbar=False)
plotting.plot_roi(destrieux['maps'])


######################################################################
# Harvard-Oxford probabilistic (4D) atlas
harvard_oxford = datasets.fetch_atlas_harvard_oxford('cort-prob-2mm')
plotting.plot_prob_atlas(harvard_oxford['maps'])


######################################################################
surf_destrieux = datasets.fetch_atlas_surf_destrieux()
fsaverage = datasets.fetch_surf_fsaverage()
plotting.view_surf(fsaverage['pial_left'], surf_destrieux['map_left'],
                   cmap='gist_ncar', colorbar=False)


######################################################################

# not needed with master
from nilearn import surface
fsaverage['sulc_left'] = surface.load_surf_data(fsaverage['sulc_left'])


######################################################################
plotting.view_surf(fsaverage['pial_left'], fsaverage['sulc_left'],
                   cmap='Greys', threshold=None, symmetric_cmap=False,
                   colorbar=False)

######################################################################
plotting.view_surf(fsaverage['infl_left'], fsaverage['sulc_left'],
                   cmap='Greys', threshold=None, symmetric_cmap=False,
                   colorbar=False)
