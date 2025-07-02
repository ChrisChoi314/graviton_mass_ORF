# graviton_mass_ORF
This is code associated with the paper that is going to be published soon. 

Authors of repository: [Chris Choi](https://github.com/ChrisChoi314) from Carnegie Mellon University 

Authors of the paper: Chris Choi and Tina Kahniashvili.

The jupyter notebook in this repository shows how all the figures in our paper were generated. Feel free to email Chris at minyeonc@andrew.cmu.edu if there are any questions about reproducing the results or generally about the paper.

The structure is something like this: 

```
graviton_mass_ORF/
├── README.md                                   ← important info about this repo
├── data/                                       ← where you should download the nanograv data into
├── src/                                        ← python scripts that reproduce the plots
│   ├── extra_functions.py                      ← function file from https://github.com/nanograv/
│   ├── fig_1.py                                ← reproduces fig 1 
│   ├── fig_2.py                                ← reproduces fig 2
│   ├── fig_3.py                                ← reproduces fig 3
│   ├── table_1.py                              ← generates the values in table 1
│   ├── rigorous_fit.py                         ← function file from https://github.com/nanograv/15yr_stochastic_analysis/tree/main/tutorials
│   ├── optimal_statistics_covariances.py       ← function file from https://github.com/nanograv/15yr_stochastic_analysis/tree/main/tutorials
│   ├── important_functions.py                  ← function file that stores functions that I made
│   └── rigorous_fit.py                         ← performs a manual least-squares fitting on the data
└──figs/                                        ← where the figs will be saved when you run the files
```

IMPORTANT: in order to run the analysis, you want to make sure you have the necessary data. Simply download this folder https://github.com/nanograv/15yr_stochastic_analysis/tree/main/tutorials/data into the folder in this repository named 'data', and make sure to rename it to 'data_pulsar'. Additionally, you need to download the folder located in the google drive mentioned in the README.md of https://github.com/nanograv/15yr_stochastic_analysis/tree/main/data_release/figure_1. Also download this into the respository named 'data' and make sure to rename it to 'figure2_data'. It should look like this:

```
graviton_mass_ORF/
├── data/                                      
│   ├── data_pulsar/                            ← contains the contents of the folder downloaded from https://github.com/nanograv/15yr_stochastic_analysis/tree/main/tutorials/data
│   └── figure2_data/                           ← containts the contents of the folder downloaded from https://drive.google.com/file/d/1zywc5zUpMSlYDrrdogPEaRd_RTr9U5J6/view?usp=sharing 
```

Note about packages: Honestly there are a ton of packages this project requires, most of them for the analysis of the NANOGrav 15-year data. Try your best to download the missing packages, but if it doesn't work, then don't worry. I have put the necessary data in the data folder, expertly exported into .npy files so that you need not do it yourself. By default, the loaded arrays are the ones being used. But it is really nice if you can get all of the packages working, makes generating figures for your paper very easy. 