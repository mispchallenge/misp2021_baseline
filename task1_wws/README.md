## MISP2021 Task 1 - Wake word spottig (WWS) baseline systems

- **Introduction**

    This is the baseline system for the Multimodal Information Based Speech Processing Challenge 2021 (MISP2021) challenge task 1. This task concerns the identification of predefined wake word(s) in utterances. ‘1’ indicates that the sample contains wake word, and ‘0’ indicates the opposite. For a more detailed description see MISP Challenge task description.


- **System description**

    The system implements a neural network (NN) based approach, where filter bank features are first extracted for each sample, and a network consisting of CNN layers, LSTM layer and fully connected layers are trained to assign labels to the audio samples.


- **Data preparation**

  - **prepare data directory**

      For training, development, and test sets, we prepare data directories by extracting the downloaded zip compressed file to the current folder. Here, we only use channel 1.

      ```
      unzip  -d ./  *.zip

      *.zip indicates the file name that needs to be unzipped
      ```

  - **speech augmentation** 

    Simulating reverberant and noisy data from near field speech, noise is widely adopted. We provide a baseline speech simulation tool to add reverberation and noise for speech augmentation. Considering that the negative samples are easier to obtain, we simulate all positive samples and partial negative samples.

    - **add reverberation**

        An open-source toolkit [pyroomacoustic](https://github.com/LCAV/pyroomacoustics) is used to add reverberation. The room impulsive response (RIR) is generated according to the actual room size and microphone position.

    - **add noise**

        We provide a simple tool to add noise with different signal-to-noise ratio. In our configuration, the reverberated speech is corrupted by the collected noise at seven signal-to-noise ratios (from -15dB to 15dB with a step of 5dB).


- **Audio Wake Word Spotting**

    For features extraction, we employ 40-dimensional filter bank (FBank) features normalized by global mean and variance as the input of the audio WWS system. The final output of the models compared with the preset threshold after sigmoid operation to calculate the false reject rate (FRR) and false alarm rate (FAR). Here comes the results.


## Results


  The system performances are as follows:
![result_readme](media/result_readme.png)


## Setting Paths

- **data prepare**

```
# Here, the given tool can simulate the positive samples directly. If you need to simulate the negative samples, you need to modify the default configuration.
--- data_prepare/run.sh ---
# Defining corpus directory
data_root=
# Defining path to python interpreter
python_path=
```

- **kws_net_only_audio**

```
--- kws_net_only_audio/run.sh ---
# Defining corpus directory
data_root=
# Defining path to python interpreter
python_path=
```

## Running the baseline audio system

- **Simulation (optional)**

```
cd data_prepare
sh run.sh
```

- **Run Training**

```
cd ../kws_net_only_audio
sh run.sh
```

## Requirments

- **pytorch**

- **python packages:**

    numpy

    tqdm

    [pyroomacoustic](https://github.com/LCAV/pyroomacoustics)

    [soundfile](https://github.com/bastibe/python-soundfile)

- **other tools:**

    [sox](http://sox.sourceforge.net/) 



