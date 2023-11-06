import pandas as pd
from jarvis.db.figshare import data as jdata



db = pd.DataFrame(jdata('mp_3d_2020'))