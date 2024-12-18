HF_REPO = Prathamesh1420/CI_CD_Pipeline_Drug_classification

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:	
	black *.py 

train:
	mkdir -p Results Model  # Create Results and Model folders if they don't exist
	python train.py
	ls -lh ./Results  # Debugging: List Results folder contents
	ls -lh ./Model    # Debugging: List Model folder contents

eval:
	echo "## Model Metrics" > report.md
	cat ./Results/metrics.txt >> report.md
	
	echo "\n## Confusion Matrix Plot" >> report.md
	echo "![Confusion Matrix](./Results/model_results.png)" >> report.md
	
	cml comment create report.md
		
update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git add ./Results ./Model  # Ensure Results and Model folders are added
	git commit -am "Update with new results"
	git push --force origin HEAD:update

hf-login: 
	pip install -U "huggingface_hub[cli]"
	git pull origin update
	git switch update
	git config --global credential.helper store  # Save the Hugging Face token persistently
	huggingface-cli login --token $(HF) --add-to-git-credential

push-hub: 
	huggingface-cli upload $(HF_REPO) ./App --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload $(HF_REPO) ./Model /Model --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload $(HF_REPO) ./Results /Metrics --repo-type=space --commit-message="Sync Model"

deploy: hf-login push-hub

all: install format train eval update-branch deploy
