# dsgs_loan_defaults
## Anaconda Environment

From the Anaconda Prompt, after navigating to the project directory, 
run the following to set up the project environment:

```
conda env create -f environment.yml
conda activate loans_env
pip install -r requirements.txt
python -m pip install -e .
```

To activate the virtual environment
```
conda activate loans_env
```

To de-activate the environment
```
conda deactivate loans_env
```