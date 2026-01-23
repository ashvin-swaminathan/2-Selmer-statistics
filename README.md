# 2-Selmer Rank Density: Cluster Setup Guide (System-Wide Install)

This guide provides step-by-step instructions to set up the environment and run the `matmethod_hyperfast_slurm.py` script on a Linux computer cluster using SLURM.

**Note:** All Python dependencies will be installed into a Conda environment named `graph_env`. The Nauty graph generator will be installed system-wide.

## Prerequisites: Install Conda

If you do not have Conda installed (and cannot load it via `module load anaconda` or `module load miniconda`), follow these steps to install Miniconda locally in your home directory.

1.  **Download the Installer:**
    ```bash
    wget [https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)
    ```

2.  **Run the Installer:**
    ```bash
    bash Miniconda3-latest-Linux-x86_64.sh
    ```
    * Press **Enter** to scroll through the license.
    * Type **yes** to accept the license.
    * Press **Enter** to accept the default install location.
    * Type **yes** when asked to initialize Minconda.

3.  **Activate Changes:**
    Close your terminal and log back in, or run:
    ```bash
    source ~/.bashrc
    ```

---

## Step 1: Set Up the Conda Environment

We will create a specific environment named `graph_env` for this project.

1.  **Create the environment** (installs Python, NumPy, and SciPy):
    ```bash
    conda create -n graph_env python=3.9 numpy scipy -y
    ```

2.  **Activate the environment:**
    *You must run this command every time you log in to work on this project.*
    ```bash
    conda activate graph_env
    ```

---

## Step 2: Install Nauty (System-Wide)

The script relies on a tool called `geng`. We will compile it and install it into `/usr/local/bin` so it is available to all users and scripts without modifying PATH variables.

**Requirement:** You must have `sudo` (root) access for this step.

1.  **Download and Extract:**
    ```bash
    wget [http://pallini.di.uniroma1.it/nauty2_8_8.tar.gz](http://pallini.di.uniroma1.it/nauty2_8_8.tar.gz)
    tar -xvzf nauty2_8_8.tar.gz
    cd nauty288
    ```

2.  **Compile:**
    ```bash
    ./configure
    make
    ```

3.  **Install System-Wide:**
    Run the install command with sudo privileges.
    ```bash
    sudo make install
    ```

4.  **Verify:**
    Check that `geng` works. You should see a help menu.
    ```bash
    geng -h
    ```
    *If this works, you can delete the `nauty288` folder if you wish.*

5.  **Return to main folder:**
    ```bash
    cd ..
    ```

---

## Step 3: Create the Python Script

1.  Open the text editor:
    ```bash
    nano matmethod_hyperfast_slurm.py
    ```

2.  **Paste the Python code** provided to you earlier into this window.

3.  **Save and Exit:**
    * Press `Ctrl + O` (then Enter) to Save.
    * Press `Ctrl + X` to Exit.

---

## Step 4: Create and Customize the SLURM Script

1.  Open the text editor:
    ```bash
    nano run_matmethod.slurm
    ```

2.  **Paste the SLURM code** provided to you earlier.

3.  **Important: Customize the Script**
    Before saving, use your arrow keys to edit the following lines in the file:

    * **Update Email:**
        Change `#SBATCH --mail-user=ashvins@math.princeton.edu` to your actual email address.
    * **Update Script Path:**
        Find the line starting with `python ...`.
        Change `/u/ashvins/Desktop/mathmethod/matmethod_hyperfast_slurm.py` to the path where you just saved your file.
        *Tip: Run `pwd` in your terminal to see your current path.*
        It should look like: `python /home/your_username/matmethod_hyperfast_slurm.py 11`
    * **Remove Resume Flag:**
        Delete `--resume 966` from the end of the line for your first run.

4.  **Save and Exit:**
    * Press `Ctrl + O` (then Enter).
    * Press `Ctrl + X`.

---

## Step 5: Run the Job

1.  **Submit the job:**
    ```bash
    sbatch run_matmethod.slurm
    ```
    *You will see a message like `Submitted batch job 12345`.*

2.  **Check status:**
    ```bash
    squeue -u $USER
    ```
    * `PD` = Pending (Waiting)
    * `R` = Running

3.  **View Output:**
    Replace `12345` with your actual Job ID:
    ```bash
    tail -f matmethod_12345.out
    ```
    *(Press `Ctrl + C` to stop watching)*
