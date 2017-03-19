# To train the model
```
./train_bayes.sh {Fraction of training data}
./train_ml.sh {Fraction of training data}
./train_map.sh {Fraction of training data}
```

# To train with cross validation
```
./train_bayes_cross_validation "{list of m0}" "{list of s0}" "{list of beta}" "{list of d}"
./train_ml_cross_validation "{list of epoch}" "{list of batch size}" "{list of learning rate}" "{list of d}"
./train_map_cross_validation "{list of epoch}" "{list of batch size}" "{list of learning rate}" "{list of alpha}" "{list of d}"

NOTE: parameter 'd' depends on the pre-preprocessing method defined in script.

```
The result of cross validation will be saved at log/{model description}
