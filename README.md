# High Gamma Dataset

This is the documentation for the High Gamma Dataset used in "Deep learning with convolutional neural networks for EEG decoding and visualization"
 (https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730).
See the paper and supporting information for a general description.

## Download
Download the files from here:
https://web.gin.g-node.org/robintibor/high-gamma-dataset
 


## Loading this dataset

The braindecode toolbox at https://github.com/robintibor/braindecode provides code to load this dataset in python.
You can run the following code to get an [MNE RawArray](https://mne-tools.github.io/stable/generated/mne.io.RawArray.html):

```python
from braindecode.datasets.bbci import  BBCIDataset
cnt = BBCIDataset(filename='./test/1.mat', load_sensor_names=None).load()
```
For using the dataset for decoding, see the next section.

## Citing
If you use this dataset in any publication, you agree to cite the above-mentioned HBM-paper as:

```
  @article {HBM:HBM23730,
  author = {Schirrmeister, Robin Tibor and Springenberg, Jost Tobias and Fiederer,
    Lukas Dominique Josef and Glasstetter, Martin and Eggensperger, Katharina and Tangermann, Michael and
    Hutter, Frank and Burgard, Wolfram and Ball, Tonio},
  title = {Deep learning with convolutional neural networks for EEG decoding and visualization},
  journal = {Human Brain Mapping},
  issn = {1097-0193},
  url = {http://dx.doi.org/10.1002/hbm.23730},
  doi = {10.1002/hbm.23730},
  month = {aug},
  year = {2017},
  keywords = {electroencephalography, EEG analysis, machine learning, end-to-end learning, brain–machine interface, 
    brain–computer interface, model interpretability, brain mapping},
  }
```

## Reproduction of our results
The `example.py` code in this repository shows how to reproduce the decoding results from the paper above and can also be used as an example code for decoding.
Please change the `data_folder` in the code to the folder where you downloaded the dataset to, see the code at the bottom of the file.
In difference to the paper, we do not use the tied neighbour loss, and we do not have biases before batch normalization layers.
You can expect the following results for this code.
They are averaged from 3 random seeds, average over all subjects.
If you need further results, please create an issue.
We show paper accuracies for comparison:

|Model|Lowpass [Hz]|Test accuracy [%]|From paper [%]|
|---|---|---|---|
|Deep|0|92.3|92.5|
|Deep|4|92.2|91.4|
|Shallow|0|88.9|89.3|
|Shallow|4|94.6|93.9|


## Data format
The data are hdf5-files, the structure is based on the structure from the Berlin Brain Computer Interface Toolbox at https://github.com/bbci/bbci_public.
Most fields have been removed, only some necessary fields are retained. We recommend to use our loading code as described above.

## Details of recording

The recodings were referenced to Cz, however in our recording setup, some residual signal remains on Cz.
Note that for subject 14, about half of the sensors lost meaningful signal in the test set.
It is still possible to get far-above chance accuracies even when not accounting for this in any way when training on all sensors of the training set.
