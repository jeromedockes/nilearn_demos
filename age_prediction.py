"""
Functional connectivity predicts age group
==========================================

This example tries to predict whether individuals are children or adults based
on different kins of functional connectivity: correlation, partial correlation,
tangent space embedding.

see `Dadi et al 2019
<https://www.sciencedirect.com/science/article/pii/S1053811919301594>`_
for a careful study.
"""
import numpy as np
import matplotlib.pylab as plt
from nilearn import datasets, input_data
from nilearn.connectome import ConnectivityMeasure
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier


######################################################################
# Load brain development fMRI dataset and MSDL atlas
# --------------------------------------------------
rest_data = datasets.fetch_development_fmri(n_subjects=30)
msdl_data = datasets.fetch_atlas_msdl()
msdl_coords = msdl_data.region_coords

######################################################################
masker = input_data.NiftiMapsMasker(
    msdl_data.maps, resampling_target="data", t_r=2, detrend=True,
    low_pass=.1, high_pass=.01, memory='nilearn_cache', memory_level=1)

pooled_subjects = []
groups = []  # child or adult
for func_file, confound_file, phenotypic in zip(
        rest_data.func, rest_data.confounds, rest_data.phenotypic):
    time_series = masker.fit_transform(func_file, confounds=confound_file)
    pooled_subjects.append(time_series)
    is_child = phenotypic['Child_Adult'] == 'child'
    groups.append(phenotypic['Child_Adult'])
_, classes = np.unique(groups, return_inverse=True)


######################################################################
# Cross-validate different ways of computing connectivity
# -------------------------------------------------------

######################################################################
# Prepare cross-validation params
kinds = ['correlation', 'partial correlation', 'tangent']
pipe = Pipeline(
    [('connectivity', 'passthrough'), ('classifier', 'passthrough')])
param_grid = [
    {'classifier': [DummyClassifier('most_frequent')]},
    {'classifier': [
        GridSearchCV(LinearSVC(), param_grid={'C': [.1, 1., 10.]}, cv=5)],
     'connectivity': [ConnectivityMeasure(vectorize=True)],
     'connectivity__kind': kinds}
]

cv = StratifiedShuffleSplit(n_splits=15, random_state=0, test_size=5)
gs = GridSearchCV(pipe, param_grid, scoring='accuracy', cv=cv, verbose=1,
                  refit=False)


######################################################################
# fit grid search
gs.fit(pooled_subjects, classes, groups=groups)
mean_scores = gs.cv_results_['mean_test_score']
scores_std = gs.cv_results_['std_test_score']

######################################################################
plt.figure(figsize=(6, 4))
positions = np.arange(len(kinds) + 1) * .1 + .1
plt.barh(positions, mean_scores, align='center', height=.05, xerr=scores_std)
yticks = ['dummy'] + list(gs.cv_results_['param_connectivity__kind'].data[1:])
yticks = [t.replace(' ', '\n') for t in yticks]
plt.yticks(positions, yticks)
plt.xlabel('Classification accuracy')
plt.gca().grid(True)
plt.gca().set_axisbelow(True)
plt.tight_layout()

plt.show()
