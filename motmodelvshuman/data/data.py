from pathlib import Path
import requests
import zipfile


LOCAL_HUMAN_RESPONSES = Path('./data/human_responses')
LOCAL_STIMULI = Path('./data/stimuli')

HUMAN_RESPONSES_URL = 'https://osf.io/download/XXXXX/' # zip file
STIMULI_URL = 'https://osf.io/download/XXXXX/' # zip file

def download_stimuli(download_again:bool=False):

    if download_again:
        LOCAL_STIMULI.rmdir()

    if not LOCAL_STIMULI.exists():
        LOCAL_STIMULI.mkdir()
        
        response = requests.get(STIMULI_URL)
        with open("stimuli.zip", "wb") as f:
            f.write(response.content)
        # unzip file to LOCAL_STIMULI folder:
        
        with zipfile.ZipFile("stimuli.zip", 'r') as zip_ref:
            zip_ref.extractall(LOCAL_STIMULI)
            
def download_human_responses(download_again:bool=False):
    
    if download_again:
        LOCAL_HUMAN_RESPONSES.rmdir()
    
    if not LOCAL_HUMAN_RESPONSES.exists():
        LOCAL_HUMAN_RESPONSES.mkdir()
        
        response = requests.get(HUMAN_RESPONSES_URL)
        with open("human_responses.zip", "wb") as f:
            f.write(response.content)
        # unzip file to LOCAL_HUMAN_RESPONSES folder:
        
        with zipfile.ZipFile("human_responses.zip", 'r') as zip_ref:
            zip_ref.extractall(LOCAL_HUMAN_RESPONSES)        
    
class Data:
    def __init__(self, experiment:str, download_again:bool=False):        
        self.experiment = experiment
        self.download_data(download_again=download_again)
        
    def download_data(self, download_again:bool=False):  
        pass

class Stimuli(Data):
    
    def download_data(self, download_again: bool = False):
        download_stimuli(download_again=download_again)
        self.data_path = LOCAL_STIMULI / self.experiment
        
class HumanResponses(Data):
    
    def download_data(self, download_again: bool = False):
        download_human_responses(download_again=download_again)
        self.data_path = LOCAL_HUMAN_RESPONSES / self.experiment


        
        