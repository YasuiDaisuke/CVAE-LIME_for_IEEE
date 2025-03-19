# README

This program is the source code for the paper published in IEEE Access: "Improving Local Fidelity and Interpretability of LIME by Replacing Only the Sampling Process with CVAE."

To run the program, you need to prepare a preprocessed dataset in CSV format by yourself.
Additionally, to compare with existing methods (e.g., LIME), you must place the XAI method programs in the `lime` directory and modify the code accordingly.

## Execution Steps

### Step1: Install Required Libraries
Install all the necessary Python libraries for this project using the following command:

```bash
pip install -r requirements.txt
```

---

### Step2: Train the Target Model and Auto Encoder by Running `train_main.py`

Run the following command to train the prediction model (to be explained) and the Conditional Variational Autoencoder (CVAE):

```bash
python train_main.py
```

After training is complete, you will be ready to proceed with the experiments.

---

### Step3: Run `test_main.py` to Obtain Experimental Results

Use the trained models to run the experiments and output the results by executing the following command:

```bash
python test_main.py
```

After execution, the experimental results will be saved in the specified save folder or file. For visualization and analysis of the results, please use the provided scripts or notebooks.

