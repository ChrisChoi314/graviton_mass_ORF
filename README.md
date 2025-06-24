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
│   ├── fig_1_compare.py                        ← reproduces fig 1 
│   ├── fig_2_nanograv.py                       ← reproduces fig 2
│   ├── fig_3_cpta.py                           ← reproduces fig 3
│   ├── table_1_chi.py                          ← generates the values in table 1
│   ├── extra_functions.py                      ← function file from https://github.com/nanograv/15yr_stochastic_analysis/tree/main/tutorials
│   ├── optimal_statistics_covariances.py       ← function file from https://github.com/nanograv/15yr_stochastic_analysis/tree/main/tutorials
│   └── polarization_func.py                    ← function file that stores functions that i made
└──figs/                                        ← where the figs will be saved when you run the files
```

If you are wondering what to download for the data, 