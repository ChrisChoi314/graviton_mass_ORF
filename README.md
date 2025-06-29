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

IMPORTANT: in order to run the analysis, you want to make sure you have the necessary data. Simply download this folder https://github.com/nanograv/15yr_stochastic_analysis/tree/main/tutorials/data into the folder in this repository named 'data'. When you run file fig_2_nanograv.py, it will generate 3 .npy files and also download them into the data folder. Then the file  table_1_chi.py will use these npy files for its analysis, so that you don't have to generate them again, which is time consuming. 

Note about packages: Honestly there are a ton of packages this project requires, most of them for the analysis of the NANOGrav 15-year data. Try your best to download the missing packages, but if it doesn't work, then don't worry. I have put the necessary data in the data folder, expertly exported into .npy files via file fig_2_nanograv.py so that you need not do it yourself (but you might still have to figure out which lines of code to delete and modify in order to get it to run without the packages, but the motivated coder should be fine!). But it is really nice if you can get all of the packages working, makes generating figures for your paper very easy. 