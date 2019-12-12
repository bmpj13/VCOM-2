## How to run

1. Download the dataset and put each folder in the corresponding one inside _lndb/_
    
    - Every folder except _folds_, _scan_cubes_ and mask_cubes_ should be filled

2. On the terminal (starting in project's root directory):

        $ pip3 install requirements.txt
        $ cd lndb/scripts
        $ python3 getNoduleCubes.py (run the others to check if everything's ok)
        $ cd ../..
        $ python3 train_i3d.py