from pathlib import Path
import shutil
from tqdm.auto import tqdm
import requests
import zipfile
import pandas as pd

LOCAL_HUMAN_RESPONSES = Path('./data/human_responses')
LOCAL_MODEL_OUTPUTS = Path('./data/model_outputs')
LOCAL_STIMULI = Path('./data/stimuli')

HUMAN_RESPONSES_URL = {'experiment1': 'https://osf.io/download/k65qe/',
                       'experiment2': 'https://osf.io/download/3kgxr/'}

MODEL_OUTPUTS_URL = {'experiment1': 'https://osf.io/download/x3bst/',
                     'experiment2': 'https://osf.io/download/e645p/'}

STIMULI_URL = {'experiment1': 'https://osf.io/download/52dt7/',
               'experiment2': 'https://osf.io/download/gt3kj/'}

def download(url, folder, experiment, download_again:bool=False):
    
    if download_again:
        shutil.rmtree(folder)

    # if folder does not exist or the folder does not contain a file or folder with a name containing "experiment1", download data:
    if not folder.exists() or not any([experiment in f.name for f in folder.iterdir()]):
        print(f"Downloading {experiment} from {url} to {folder}...")
        
        folder.mkdir(parents=True, exist_ok=True)
        
        # download zip file
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
        download(self.url[experiment], self.folder, self.experiment, download_again=download_again)
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
        self.data_path = self.folder / f"{self.experiment}.csv"
        self.data = pd.read_csv(self.data_path, index_col=[0,1])
        
class ModelOutput(Data):
    url = MODEL_OUTPUTS_URL
    folder = LOCAL_MODEL_OUTPUTS
    def __init__(self, experiment:str, model_name:str, additional_model_id:str=None, download_again:bool=False):        
        self.experiment = experiment
        self.model_name = model_name
        self.additional_model_id = additional_model_id
        download(self.url[experiment], self.folder, self.experiment, download_again=download_again)
        self.read_data()    
    def read_data(self):
        if self.additional_model_id:
            # e.g., additional_model_id = 'noisy-reid_1.55'
            self.data_path = self.folder / self.experiment / f"{self.model_name}_{self.experiment}_{self.additional_model_id}.pkl"
        else:
            self.data_path = self.folder / self.experiment / f"{self.model_name}_{self.experiment}.pkl"
        self.data = pd.read_pickle(self.data_path)
                
        