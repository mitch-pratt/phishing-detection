import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_csv(self):
        self.df = pd.read_csv(
            self.file_path,
            encoding="latin-1",
            engine="python",
            on_bad_lines="skip"
        )
        return self.df
    
    def clean(self, label_column="label"):
        self.df = self.df.dropna(subset=[label_column])
        return self.df
    
    def get_urls_labels(self, url_column="domain", label_column="label"):
        urls = self.df[url_column].astype(str).tolist()
        labels = self.df[label_column].astype(int).tolist()
        return urls, labels
    
    #function to check how many bad lines?
    #function to drop missed URLs?