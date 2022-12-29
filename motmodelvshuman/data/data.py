from pathlib import Path
import shutil
from tqdm.auto import tqdm
import requests
import zipfile
import pandas as pd

LOCAL_HUMAN_RESPONSES = Path('./data/human_responses')
LOCAL_MODEL_OUTPUTS = Path('./data/model_outputs')
LOCAL_STIMULI = Path('./data/stimuli')

HUMAN_RESPONSES_URL = {'experiment1': 'https://osf.io/download/XXXXX/',
                       'experiment2': 'https://osf.io/download/XXXXX/'}

MODEL_OUTPUTS_URL = {'experiment1': 'https://osf.io/download/ws9cd/',
                     'experiment2': 'https://osf.io/download/kswbx/'}

STIMULI_URL = {'experiment1': 'https://osf.io/download/XXXXX/',
               'experiment2': 'https://osf.io/download/XXXXX/'}

def download(url, folder, download_again:bool=False):
    
    if download_again:
        shutil.rmtree(folder)

    if not folder.exists():
        print(f"Downloading data from {url} to {folder}...")
        
        folder.mkdir(parents=True, exist_ok=True)
        
        # download zip file
        print(url)
        response = requests.get(url, stream=True)
        with open("data.zip", "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size=1024), total=int(response.headers.get('Content-Length', 0))/1024):
                f.write(chunk)
            
        # unzip file to LOCAL_HUMAN_RESPONSES folder
        with zipfile.ZipFile("data.zip", 'r') as zip_ref:
            zip_ref.extractall(folder)
            
        # delete zip file
        Path('data.zip').unlink()    
    
class Data:
    url = None
    folder = None    
    def __init__(self, experiment:str, download_again:bool=False):        
        self.experiment = experiment
        download(self.url[experiment], self.folder, download_again=download_again)
        self.read_data()
    def read_data(self):
        pass
        
        
class Stimuli(Data):
    url = STIMULI_URL
    folder = LOCAL_STIMULI
    def read_data(self):
        self.data_path = self.folder / self.experiment
        self.annotations_json_path = self.data_path / 'annotations' / 'train_cocoformat.json'
        files = [f for f in (self.data_path / 'experimental_sessions').glob('**/*.json') if f.is_file()]
        self.experimental_session_id = [f.stem for f in files][0]
        self.experimental_session_file = self.data_path / 'experimental_sessions' / f"{self.experimental_session_id}.json"
class HumanResponses(Data):
    url = HUMAN_RESPONSES_URL
    folder = LOCAL_HUMAN_RESPONSES
    def read_data(self):
        self.data_path = self.folder / self.experiment / f"{self.experiment}_responses.csv"
        self.data = pd.read_csv(self.data_path, index_col=[0,1])
        
class ModelOutput(Data):
    url = MODEL_OUTPUTS_URL
    folder = LOCAL_MODEL_OUTPUTS
    def __init__(self, experiment:str, model_name:str, download_again:bool=False):        
        self.experiment = experiment
        self.model_name = model_name
        download(self.url[experiment], self.folder, download_again=download_again)
        self.read_data()    
    def read_data(self):
        self.data_path = self.folder / self.experiment / f"{self.model_name}_{self.experiment}.pkl"
        self.data = pd.read_pickle(self.data_path)
                
        