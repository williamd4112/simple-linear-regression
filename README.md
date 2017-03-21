# Introduction
Python Tensorflow implementation for three kinds of linear regression algorithm. (Maximum Likelihood, Maximum a posterior, Bayesian). This project aims to predict height map of south Taiwan and study the difference of these three kinds of linear regression algorithms. (Implementation detail mentioned in doc/report.pdf)

# Results
Minimumn Mean-square error (MSE) of three approaches

| ML      | MAP          | Bayesian  |
| ------------- |:-------------:| -----:|
| 90.525   | 88.585709 | 60.591061|

# Visualization
## Maximum Likelihood
| 3D            | 2D           |
| ------------- |:------------:|
|![ml-3d](/doc/ml-3d.png)|![ml-2d](/doc/ml-2d.png)|

## Maximum a Posterior
| 3D            | 2D           |
| ------------- |:------------:|
|![map-3d](/doc/map-3d.png)|![map-2d](/doc/map-2d.png)|

## Bayesian
| 3D            | 2D           |
| ------------- |:------------:|
|![bayes-3d](/doc/bayes-3d.png)|![bayes-2d](/doc/bayes-2d.png)|


# Dependencies
- numpy
- Tensorflow
- Scipy (kmeans)
- Scikit-learn (k-fold)

# To run pre-trained model
```
./test_bayes.sh {X} model/bayes/bayes.npy model/bayes/bayes-mean.npy model/bayes/bayes-sigma.npy {Y}
./test_ml.sh {X} model/ml/ml.npy model/ml/ml-mean.npy model/ml/ml-sigma.npy {Y}
./test_map.sh {X} model/map/map.npy model/map/map-mean.npy model/map/map-sigma.npy {Y}

```
Prediction results would be saved at {Y} (output path)

# To score the predictions
```
python score.py {predictions}.csv {ground truth}.csv
```

# To train the model
```
./train_bayes.sh {Fraction of training data}
./train_ml.sh {Fraction of training data}
./train_map.sh {Fraction of training data}

All hyperparameters in the scripts are set to optimal settings.

```

# To train with cross validation
```
./train_bayes_cross_validation.sh "{list of m0}" "{list of s0}" "{list of beta}" "{list of d}"
./train_ml_cross_validation.sh "{list of epoch}" "{list of batch size}" "{list of learning rate}" "{list of d}"
./train_map_cross_validation.sh "{list of epoch}" "{list of batch size}" "{list of learning rate}" "{list of d}" "{list of alpha}"

NOTE: parameter 'd' depends on the pre-preprocessing method defined in script.
pre=grid: grid cell size
pre=kmeans: number of cluster

e.g. ./train_bayes_cross_validation.sh "0.0" "2.0" "25.0 12.5" "1024 2048"
```
The result of cross validation will be saved at log/{model description}

# To test the model
```
./test_bayes.sh {input data X} {model path} {model mean path} {model sigma path} {output path}
./test_ml.sh {input data X} {model path} {model mean path} {model sigma path} {output path}
./test_map.sh {input data X} {model path} {model mean path} {model sigma path} {output path}

e.g. ./test_bayes.sh X_test.csv model/bayes-m0-0.0-s0-2.0-beta-25.0-grid-0.015.npy model/bayes-m0-0.0-s0-2.0-beta-25.0-grid-0.015-mean.npy model/bayes-m0-0.0-s0-2.0-beta-25.0-grid-0.015-sigma.npy
```
