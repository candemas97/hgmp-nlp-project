import pandas as pd
from transformers import RobertaTokenizer
from torch.utils.data import Dataset


class SongDataset(Dataset):
    def __init__(self, filename, tokenizer_name, max_length=512):
        # Cargar el archivo CSV
        self.data = pd.read_csv(filename)

        # Inicializar el tokenizador
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extraer la fila correspondiente
        row = self.data.iloc[idx]

        # Tokenizar la columna "Cancion"
        inputs = self.tokenizer.encode_plus(row["normalize_2"], add_special_tokens=True, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

        # Convertir género a entero (suponiendo que es una clasificación binaria)
        label = int(row["Genero"])

        # Retorna un diccionario con los inputs y la etiqueta
        return {"input_ids": inputs["input_ids"].squeeze(), "attention_mask": inputs["attention_mask"].squeeze(), "label": label}


# Si quieres realizar una prueba rápida del dataset:
if __name__ == "__main__":
    # Crear una instancia del dataset
    dataset = SongDataset("../data/train.csv", "roberta-base")

    # Obtener el primer elemento y visualizarlo
    sample = dataset[0]
    print(sample)
