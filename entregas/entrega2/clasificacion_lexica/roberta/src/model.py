import torch
import torch.nn as nn
from transformers import RobertaModel


class SongClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super(SongClassifier, self).__init__()

        # Cargar el modelo base de RoBERTa
        self.roberta = RobertaModel.from_pretrained(model_name)

        # Capa de clasificación
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        # Extraer los features del modelo RoBERTa
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        # Tomar la salida del token [CLS] como representación de la secuencia
        sequence_output = outputs.last_hidden_state[:, 0, :]

        # Pasar la salida a través de la capa de clasificación
        logits = self.classifier(sequence_output)

        return logits


# Función auxiliar para inicializar el modelo y enviarlo a la GPU si está disponible
def initialize_model(model_name, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SongClassifier(model_name, num_classes)
    model = model.to(device)

    return model, device


# Si quieres realizar una prueba rápida del modelo:
if __name__ == "__main__":
    model, _ = initialize_model("roberta-base", 2)  # Suponiendo clasificación binaria
    print(model)
