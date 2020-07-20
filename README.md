# CP-VAE

On Variational Learning of Controllable Representations for Text without Supervision

Paper published in ICML 2020: [arXiv](https://arxiv.org/abs/1905.11975)

### Preprocess

Download the GloVe embeddings from [this link](http://nlp.stanford.edu/data/glove.840B.300d.zip) to `data/`. Then run:

```
python preprocess.py --data_name <data_name>
```

Data name can be either *yelp* or *amazon*.

### Train

*Vanilla VAE*:

```
python run_baseline.py --data_name <data_name>
```

Hyper-parameters can be set in `baseline_config.py`.


*CP-VAE*:

```
python run.py --data_name <data_name>
```

Hyper-parameters can be set in `config.py`.

### Evaluation

Train the sentiment classifier:

```
python classifier.py --data_name <data_name>
```

Perform unsupervised style transfer:

*Vanilla VAE*:

```
python transfer_baseline.py --data_name <data_name> --load_path <path_to_checkpoint> --type <magnitude_type>
```

Magnitude type can be:

- 0, no change
- 1, move by one std
- 2, move by two std
- 3, move to extremum

*CP-VAE*:

```
python transfer.py --data_name <data_name> --load_path <path_to_checkpoint>
```

Calculate sentiment classification accuracy and BLEU score against the source sentence:

```
python evaluate.py --data_name <data_name> --target_path <path_to_target>
```

### Analysis

*Visualization of NLL discrepency*

Copy the nll statistics from the checkpoint folders to `plot/` and then run:

```
python plot_nll.py
```

*Topological Data Analysis*

```
python tda.py --data_name <data_name> --load_path <path_to_checkpoint> --resolution <n>
```

```
python tda_baseline.py --data_name <data_name> --load_path <path_to_checkpoint> --resolution <n>
```

### Cite

If you found this codebase or our work useful, please cite:

```
@InProceedings{xu2020variational,
    author={Xu, Peng and Cheung, Jackie Chi Kit and Cao, Yanshuai},
    title={On Variational Learning of Controllable Representations for Text without Supervision},
    booktitle={The 37th International Conference on Machine Learning (ICML 2020)},
    month={July},
    year={2020},
    publisher={PMLR},
}
```

### License

Copyright (c) 2018-present, Royal Bank of Canada. All rights reserved.
This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.
