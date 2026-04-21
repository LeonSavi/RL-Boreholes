# RL-Boreholes


### Setup

- Install requirements:

```
pip install -r requirements.txt

```

- Run scraper.py for nlog data. 
- Download LILY data from [zenodo](https://zenodo.org/records/10425539) and add them in /data/lily/. To download the following .csv files: KAPPA_DataLITH, MAD_DataLITH, NGR_DataLITH, PWC_DataLITH, TCON_DataLITH 
- Run pull_data.py -> You will get a parquet file with combined dataset
- Run analysis for charts