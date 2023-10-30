import pandas as pd
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset

class SongDataset(Dataset):
    def __init__(self, filename, tokenizer_name, max_length=512):
        # Cargar el archivo CSV
        self.data = pd.read_csv(filename)

        # Inicializar el tokenizador de GPT-2
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)

        # Agregar un nuevo token de relleno al vocabulario
        special_tokens_dict = {'pad_token': '[PAD]'}
        self.tokenizer.add_special_tokens(special_tokens_dict)

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

if __name__ == "__main__":
    # Crea una instancia de SongDataset
    dataset = SongDataset("../data/train.csv", "gpt2-medium")

    # Muestra el tamaño del dataset
    print(f"Total samples: {len(dataset)}")

    # Muestra un ejemplo (la primera entrada)
    sample = dataset[0]
    print(f"Input IDs: {sample['input_ids']}")
    print(f"Attention Mask: {sample['attention_mask']}")
    print(f"Label: {sample['label']}")
