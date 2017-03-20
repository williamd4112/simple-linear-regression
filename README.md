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
