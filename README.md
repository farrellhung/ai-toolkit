Original repo: https://github.com/ostris/ai-toolkit
Forked for fine-tuning research.

### Setup
```bash
git clone https://github.com/farrellhung/ai-toolkit.git
cd ai-toolkit
git submodule update --init --recursive
python -m venv venv
source venv/bin/activate
pip install torch
pip install -r requirements.txt
pip install --upgrade accelerate transformers diffusers huggingface_hub #Optional, run it if you run into issues
huggingface-cli login #For Flux Dev
```
### Train
```bash
python run.py config/my_config.yml
```