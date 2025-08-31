class Mapper:
    def __init__(self, metadata):
        self.metadata = metadata

    def process(self, df):
        for col in df.columns:
            if col + "_int" in self.metadata:
                mapping = {string: num for string, num in zip(self.metadata[col]["unique"], self.metadata[col + "_int"]["unique"])}
                df[col] = df[col].astype(str).map(mapping, na_action='ignore')

        df.dropna(inplace=True)
        for col in df.columns:
            if self.metadata[col]["type"] in ["int", "str"]:
                df[col] = df[col].astype(int)

        return df
    
    def restore(self, df):
        for col in df.columns:
            if col + "_int" in self.metadata:
                mapping = {num: string for string, num in zip(self.metadata[col]["unique"], self.metadata[col + "_int"]["unique"])}
                df[col] = df[col].map(mapping, na_action='ignore')
        return df
