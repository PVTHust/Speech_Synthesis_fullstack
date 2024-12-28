# model TTS
## Pre-requisites
1. Python >= 3.7
2. Clone this repository:
```bash
git clone https://github.com/yl4579/StyleTTS2.git
cd StyleTTS2
```
3. Install python requirements: 
```bash
pip install -r requirements.txt
```
On Windows add:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -U
```
Also install phonemizer and espeak if you want to run the demo:
```bash
pip install phonemizer
sudo apt-get install espeak-ng
```
4. Download and extract the [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/), unzip to the data folder and upsample the data to 24 kHz. The text aligner and pitch extractor are pre-trained on 24 kHz data, but you can easily change the preprocessing and re-train them using your own preprocessing. 
For LibriTTS, you will need to combine train-clean-360 with train-clean-100 and rename the folder train-clean-460 (see [val_list_libritts.txt](https://github.com/yl4579/StyleTTS/blob/main/Data/val_list_libritts.txt) as an example).
## Training
First stage training:
```bash
accelerate launch train_first.py --config_path ./Configs/config.yml
```
Second stage training **(DDP version not working, so the current version uses DP, again see [#7](https://github.com/yl4579/StyleTTS2/issues/7) if you want to help)**:
```bash
python train_second.py --config_path ./Configs/config.yml
```
You can run both consecutively and it will train both the first and second stages. The model will be saved in the format "epoch_1st_%05d.pth" and "epoch_2nd_%05d.pth". Checkpoints and Tensorboard logs will be saved at `log_dir`. 

The data list format needs to be `filename.wav|transcription|speaker`, see [val_list.txt](https://github.com/yl4579/StyleTTS2/blob/main/Data/val_list.txt) as an example. The speaker labels are needed for multi-speaker models because we need to sample reference audio for style diffusion model training. 

### Important Configurations
In [config.yml](https://github.com/yl4579/StyleTTS2/blob/main/Configs/config.yml), there are a few important configurations to take care of:
- `OOD_data`: The path for out-of-distribution texts for SLM adversarial training. The format should be `text|anything`.
- `min_length`: Minimum length of OOD texts for training. This is to make sure the synthesized speech has a minimum length.
- `max_len`: Maximum length of audio for training. The unit is frame. Since the default hop size is 300, one frame is approximately `300 / 24000` (0.0125) second. Lowering this if you encounter the out-of-memory issue. 
- `multispeaker`: Set to true if you want to train a multispeaker model. This is needed because the architecture of the denoiser is different for single and multispeaker models.
- `batch_percentage`: This is to make sure during SLM adversarial training there are no out-of-memory (OOM) issues. If you encounter OOM problem, please set a lower number for this. 

### Pre-trained modules
In [Utils](https://github.com/yl4579/StyleTTS2/tree/main/Utils) folder, there are three pre-trained models: 
- **[ASR](https://github.com/yl4579/StyleTTS2/tree/main/Utils/ASR) folder**: It contains the pre-trained text aligner, which was pre-trained on English (LibriTTS), Japanese (JVS), and Chinese (AiShell) corpus. It works well for most other languages without fine-tuning, but you can always train your own text aligner with the code here: [yl4579/AuxiliaryASR](https://github.com/yl4579/AuxiliaryASR).
- **[JDC](https://github.com/yl4579/StyleTTS2/tree/main/Utils/JDC) folder**: It contains the pre-trained pitch extractor, which was pre-trained on English (LibriTTS) corpus only. However, it works well for other languages too because F0 is independent of language. If you want to train on singing corpus, it is recommended to train a new pitch extractor with the code here: [yl4579/PitchExtractor](https://github.com/yl4579/PitchExtractor).
- **[PLBERT](https://github.com/yl4579/StyleTTS2/tree/main/Utils/PLBERT) folder**: It contains the pre-trained [PL-BERT](https://arxiv.org/abs/2301.08810) model, which was pre-trained on English (Wikipedia) corpus only. It probably does not work very well on other languages, so you will need to train a different PL-BERT for different languages using the repo here: [yl4579/PL-BERT](https://github.com/yl4579/PL-BERT). You can also use the [multilingual PL-BERT](https://huggingface.co/papercup-ai/multilingual-pl-bert) which supports 14 languages. 

### Common Issues
- **Loss becomes NaN**: If it is the first stage, please make sure you do not use mixed precision, as it can cause loss becoming NaN for some particular datasets when the batch size is not set properly (need to be more than 16 to work well). For the second stage, please also experiment with different batch sizes, with higher batch sizes being more likely to cause NaN loss values. We recommend the batch size to be 16. You can refer to issues [#10](https://github.com/yl4579/StyleTTS2/issues/10) and [#11](https://github.com/yl4579/StyleTTS2/issues/11) for more details.
- **Out of memory**: Please either use lower `batch_size` or `max_len`. You may refer to issue [#10](https://github.com/yl4579/StyleTTS2/issues/10) for more information.
- **Non-English dataset**: You can train on any language you want, but you will need to use a pre-trained PL-BERT model for that language. We have a pre-trained [multilingual PL-BERT](https://huggingface.co/papercup-ai/multilingual-pl-bert) that supports 14 languages. You may refer to [yl4579/StyleTTS#10](https://github.com/yl4579/StyleTTS/issues/10) and [#70](https://github.com/yl4579/StyleTTS2/issues/70) for some examples to train on Chinese datasets. 

## Finetuning
The script is modified from `train_second.py` which uses DP, as DDP does not work for `train_second.py`. Please see the bold section above if you are willing to help with this problem. 
```bash
python train_finetune.py --config_path ./Configs/config_ft.yml
```
Please make sure you have the LibriTTS checkpoint downloaded and unzipped under the folder. The default configuration `config_ft.yml` finetunes on LJSpeech with 1 hour of speech data (around 1k samples) for 50 epochs. This took about 4 hours to finish on four NVidia A100. The quality is slightly worse (similar to NaturalSpeech on LJSpeech) than LJSpeech model trained from scratch with 24 hours of speech data, which took around 2.5 days to finish on four A100. The samples can be found at [#65 (comment)](https://github.com/yl4579/StyleTTS2/discussions/65#discussioncomment-7668393). 

If you are using a **single GPU** (because the script doesn't work with DDP) and want to save training speed and VRAM, you can do (thank [@korakoe](https://github.com/korakoe) for making the script at [#100](https://github.com/yl4579/StyleTTS2/pull/100)):
```bash
accelerate launch --mixed_precision=fp16 --num_processes=1 train_finetune_accelerate.py --config_path ./Configs/config_ft.yml
```
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yl4579/StyleTTS2/blob/main/Colab/StyleTTS2_Finetune_Demo.ipynb)


## Inference
Please refer to [Inference_LJSpeech.ipynb](https://github.com/yl4579/StyleTTS2/blob/main/Demo/Inference_LJSpeech.ipynb) (single-speaker) and [Inference_LibriTTS.ipynb](https://github.com/yl4579/StyleTTS2/blob/main/Demo/Inference_LibriTTS.ipynb) (multi-speaker) for details. For LibriTTS, you will also need to download [reference_audio.zip](https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/reference_audio.zip) and unzip it under the `demo` before running the demo. 

- The pretrained StyleTTS 2 on LJSpeech corpus in 24 kHz can be downloaded at [https://huggingface.co/yl4579/StyleTTS2-LJSpeech/tree/main](https://huggingface.co/yl4579/StyleTTS2-LJSpeech/tree/main).

  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yl4579/StyleTTS2/blob/main/Colab/StyleTTS2_Demo_LJSpeech.ipynb)

- The pretrained StyleTTS 2 model on LibriTTS can be downloaded at [https://huggingface.co/yl4579/StyleTTS2-LibriTTS/tree/main](https://huggingface.co/yl4579/StyleTTS2-LibriTTS/tree/main). 

  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yl4579/StyleTTS2/blob/main/Colab/StyleTTS2_Demo_LibriTTS.ipynb)


You can import StyleTTS 2 and run it in your own code. However, the inference depends on a GPL-licensed package, so it is not included directly in this repository. A [GPL-licensed fork](https://github.com/NeuralVox/StyleTTS2) has an importable script, as well as an experimental streaming API, etc. A [fully MIT-licensed package](https://pypi.org/project/styletts2/) that uses gruut (albeit lower quality due to mismatch between phonemizer and gruut) is also available.  

***Before using these pre-trained models, you agree to inform the listeners that the speech samples are synthesized by the pre-trained models, unless you have the permission to use the voice you synthesize. That is, you agree to only use voices whose speakers grant the permission to have their voice cloned, either directly or by license before making synthesized voices public, or you have to publicly announce that these voices are synthesized if you do not have the permission to use these voices.*** 

## References
- [archinetai/audio-diffusion-pytorch](https://github.com/archinetai/audio-diffusion-pytorch)
- [jik876/hifi-gan](https://github.com/jik876/hifi-gan)
- [rishikksh20/iSTFTNet-pytorch](https://github.com/rishikksh20/iSTFTNet-pytorch)
- [nii-yamagishilab/project-NN-Pytorch-scripts/project/01-nsf](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/project/01-nsf)


# DEMO WebUI
## StyleTTS WebUI
An all-in-one inferencing and training WebUI for StyleTTS.  The intended compatbility is meant for Windows, but should still work with a little bit of modification for WSL or Linux.
> StyleTTS actually trains nicer in WSL than windows, so I might add compatibiltiy here sometime in the future.

## Features
✔️ Inferencing/Generation Tab with ability to choose between different trained models

✔️ Split and combine paragraphs to generate audio for text of any arbitrary length

✔️ Dataset prepration using Whisperx

✔️ Training tab with tensorboard monitoring available

## Setup
There is no Linux or Mac set-up at the moment. However, I think the set-up on linux isn't too convoluted as it doesn't require any code modifications, just installation modifications.  I believe you do not need to uninstall and reinstall torch and then the back slashes should be replaced with forward slashes in the commands below.

### Windows Package
Is available for Youtube Channel Members at the Supporter (Package) level: https://www.youtube.com/channel/UCwNdsF7ZXOlrTKhSoGJPnlQ/join

**Minimum Requirements**
- Nvidia Graphics Card (12GB VRAM is the minimum recommendation for training at a decent speed, 8GB possible though, albeit very slow. See below troubleshooting for more information)
- Windows 10/11
1. After downloading the zip file, unzip it.
2. Launch the webui with launch_webui.bat

### Manual Installation (Windows only)
**Prerequisites**
- Python 3.11: https://www.python.org/downloads/release/python-3119/
- git cmd tool: https://git-scm.com/
- vscode or some other IDE (optional)
- Nvidia Graphics Card (12GB VRAM is the minimum recommendation for training at a decent speed, 8GB possible though, albeit very slow. See below troubleshooting for more information)
- Microsoft build tools, follow: https://stackoverflow.com/questions/64261546/how-to-solve-error-microsoft-visual-c-14-0-or-greater-is-required-when-inst/64262038#64262038

0. Install FFMPEG, overall, just a good tool to have and is needed for the repo.
    - https://www.youtube.com/watch?v=JR36oH35Fgg&ab_channel=Koolac
1. Clone the repository
```
git clone https://github.com/JarodMica/StyleTTS-WebUI.git
```
2. Navigate into the repo
```
cd .\StyleTTS-WebUI\
```
3. Setup a virtual environement, specifying python 3.11
```
py -3.11 -m venv venv
```
4. Activate venv.  If you've never run venv before on windows powershell, you will need to change ExecutionPolicy to RemoteSigned
```
.\venv\Scripts\activate
```
5. Install torch manually as windows does not particularly like just installing torch, you need to install prebuilt wheels.
> **NOTE:** torch installed with 2.4.0 or higher was causing issues with cudnn and cublas dlls not being found (presumed due to ctranslate2).  Make sure you use 2.3.1 as specified in the command below.
```
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```
6. Run the requirements.txt (Before this, make sure you have microsoft build tools installed, else, it will fail for some packages)
```
pip install -r .\requirements.txt
```
  - 6.1. Check torch, if it is greater than 2.3.1, uninstall and reinstall, else, you can continue on and no need to run the below:
    ```
    pip show torch
    ```
    If greater than 2.3.1, uninstall and reinstall:
    ```
    pip uninstall torch
    pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
    ```
7. Initialize submodules in the repository
```
git submodule init
git submodule update --remote
```
8. Install the StyleTTS2 package into venv
```
pip install .\modules\StyleTTS2\
```
9. Download the pretrained StyleTTS2 Model and yaml here:https://huggingface.co/yl4579/StyleTTS2-LibriTTS/tree/main/Models/LibriTTS.  You'll need to place them into the folder ```pretrain_base_1``` inside of the ```models``` folder.  The file structure should look like the below.
```
models\pretrain_base_1\epochs_2nd_00020.pth
models\pretrain_base_1\config.yml
```
10. Install eSpeak-NG onto your computer.  Head over to https://github.com/espeak-ng/espeak-ng/releases and select the ```espeak-ng-X64.msi``` the assets dropdown.  Download, run, and follow the prompts to set it up on your device.  As of this write-up, it'll be at the bottom of 1.51 on the github releases page
> You can remove the program by going to "Add or remove programs" on your computer, then searching for espeak.
11. Download punkt by running the below python script:
```
python .\modules\StyleTTS2\styletts2\download_punkt.py
```
12.. Run the StyleTTS2 Webui
```
python webui.py
```
13. (Optional) Make a .bat file to automatically run the webui.py each time without having to activate venv each time. How to: https://www.windowscentral.com/how-create-and-run-batch-file-windows-10
```
call venv\Scripts\activate
python webui.py
```

## Usage
There are 3 Tabs: Generation, Training, and Settings

### Generation
Before you start generating, you need a small reference audio file (preferably wave file) to generate style vectors from.  This can be used for "zero shot" cloning as well, but you'll do the same thing for generating after training a model.

To do this, go into the ```voices``` folder, then create a new folder and name it whatever speaker name you'd like.  Then, place the small reference audio file into that folder.  The full path should look like below:
```
voices/name_of_your_speaker/reference_audio.wav
```
If you had already launched the webui, click on the ```Update Voices``` button and it'll update the voices that are now available to choose from.

One thing to note is the ```Settings``` tab contains the StyleTTS models that are available, but by default, if no training has been done, the base pretrained model will be selected.  After training, you'll be able to change what model is loaded.

|Field      |Description|
|-----------|-----------|
|Input text| The text you want to generate |
|Voice| Voices that are available |
|Reference Audio| The audio file to use as a reference for generation|
|Seed| A number randomly assigned to each generation.  A seed will generate the same audio output no matter how many times you generate.  Set to -1 to have it be randomized|
|alpha| Affects speaker timbre, the higher the value, the further it is from the reference sample. At 0, may sound closer to reference sample at the cost of a little quality|
|beta| Affects speaker prosody and expressiveness.  The higher the value, the more exaggerated speech may be.|
|Diffusion Steps| Affects quality at the cost of some speed.  The higher the number, the more denoising-steps are done (in relation to diffusion models not audio noise)|
|Embedding Scale| Affects speaker expressiveness/emotion.  A higher value may result in higher emotion or expression.|

