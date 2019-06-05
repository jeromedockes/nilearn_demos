"""
What's new in Nilearn
=====================

Nilearn can do many things. We arbitrarily picked
a few for this 5-minute demo. Check out the documentation
(https://nilearn.github.io/user_guide.html) and the example gallery
(https://nilearn.github.io/auto_examples/index.html) for more!

"""

######################################################################
import warnings
warnings.simplefilter('ignore')

######################################################################
# Interactive image vizualization: brain volume and cortical surface
# ------------------------------------------------------------------

# Download and plot a group-level statistical map:
from nilearn import datasets, plotting

img = datasets.fetch_neurovault_motor_task()['images'][0]
plotting.view_img(img, threshold='95%')

######################################################################
view = plotting.view_img(img, threshold='95%')
view.open_in_browser()

######################################################################
# Static plots (matplotlib figures)

plotting.plot_stat_map(
    img, threshold=3, display_mode='z', cut_coords=5)

######################################################################
# Projections on the cortical surface

plotting.view_img_on_surf(img, threshold='95%')

######################################################################
# More about dataset downloaders: https://nilearn.github.io/modules/reference.html#module-nilearn.datasets
#
# More about plotting: https://nilearn.github.io/plotting/index.html


###############################################################################
# Easily creating connectivity matrices
# -------------------------------------
# we use one subject from a dataset recently added to Nilearn:
# a movie watching based brain development dataset

rest_data = datasets.fetch_development_fmri(n_subjects=1)
rest_data.keys()

###############################################################################
# Extract time series from probabilistic ROIs of the MSDL atlas.
from nilearn.input_data import NiftiMapsMasker

msdl = datasets.fetch_atlas_msdl()
print('number of regions in MSDL atlas:', len(msdl.labels))

masker = NiftiMapsMasker(
    msdl.maps, resampling_target="data", t_r=2, detrend=True,
    low_pass=.1, high_pass=.01, memory='nilearn_cache').fit([])
masked_data = masker.transform(rest_data.func[0], rest_data.confounds[0])
print('masked data shape:', masked_data.shape)

###############################################################################
# Compute and plot connectivity matrix

from nilearn.connectome import ConnectivityMeasure

correlation = ConnectivityMeasure(
    kind='correlation').fit_transform([masked_data])[0]

plotting.plot_matrix(correlation, tri='lower')

###############################################################################
plotting.view_connectome(
    correlation, msdl.region_coords, threshold='90%', cmap='cold_hot')

###############################################################################
# `ConnectivityMeasure` can be used to extract features for supervised learning

connectivity = ConnectivityMeasure(vectorize=True).fit_transform(
    [masked_data, masked_data, masked_data])
print('(n samples, n features):', connectivity.shape)

###############################################################################
# More about connectivity with nilearn: https://nilearn.github.io/connectivity/index.html
#
# More about the development dataset:
# Richardson et al. (2018). Development of the social brain from age three to
# twelve years.
#
# Using this dataset and connectivity to classify adults vs children: https://github.com/nilearn/nilearn/blob/master/examples/03_connectivity/plot_group_level_connectivity.py
