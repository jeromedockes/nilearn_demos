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
from nilearn import datasets

img = datasets.fetch_neurovault_motor_task()['images'][0]
print(img)

######################################################################
from nilearn import plotting

plotting.view_img(img, threshold='95%')

######################################################################
# Made with the brainsprite viewer: https://github.com/SIMEXP/brainsprite.js

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

plotting.view_img_on_surf(img, threshold='95%', surf_mesh='fsaverage')

######################################################################
# More about dataset downloaders: https://nilearn.github.io/modules/reference.html#module-nilearn.datasets
#
# More about plotting: https://nilearn.github.io/plotting/index.html


###############################################################################
# Easily creating connectivity matrices
# -------------------------------------
# we use one subject from a dataset recently added to Nilearn:
# a movie watching based brain development dataset

rest_data = datasets.fetch_development_fmri(n_subjects=60)
rest_data.keys()

###############################################################################
# Extract time series from probabilistic ROIs of the MSDL atlas.
from nilearn.input_data import NiftiMapsMasker
import numpy as np

msdl = datasets.fetch_atlas_msdl()
print('number of regions in MSDL atlas:', len(msdl.labels))

masker = NiftiMapsMasker(
    msdl.maps, resampling_target="data", t_r=2, detrend=True,
    low_pass=.1, high_pass=.01, memory='nilearn_cache', memory_level=3).fit([])
masked_data = [masker.transform(func, confounds) for
               (func, confounds) in zip(rest_data.func, rest_data.confounds)]
masked_data = np.asarray(masked_data)
print('masked data shape:', masked_data[0].shape)

###############################################################################
# Compute and plot connectivity matrix

from nilearn.connectome import ConnectivityMeasure

correlation_measure = ConnectivityMeasure(kind='correlation').fit(masked_data)

plotting.plot_matrix(correlation_measure.mean_, tri='lower')

###############################################################################
plotting.view_connectome(
    correlation_measure.mean_, msdl.region_coords,
    threshold='90%', cmap='cold_hot')

###############################################################################
# Age group classification with scikit-learn
# ------------------------------------------
# `ConnectivityMeasure` can be used to extract features for supervised learning
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

kinds = ['correlation', 'partial correlation', 'tangent']
groups = [pheno['Child_Adult'] for pheno in rest_data.phenotypic]
classes = LabelEncoder().fit_transform(groups)

cv = StratifiedShuffleSplit(n_splits=15, random_state=0, test_size=10)

correlations = ConnectivityMeasure(
    kind='correlation', vectorize=True).fit_transform(masked_data)

scores = cross_val_score(LinearSVC(), correlations, classes, cv=cv)
print(np.mean(scores))

###############################################################################
import seaborn as sns
sns.violinplot(scores)
sns.stripplot(scores, color='k')

###############################################################################
# More about connectivity with nilearn: https://nilearn.github.io/connectivity/index.html
#
# More about the development dataset:
# Richardson et al. (2018). Development of the social brain from age three to
# twelve years.
