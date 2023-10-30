import torch


def train(model, dataloader, optimizer, criterion, device):
    """
    Realiza una época de entrenamiento.

    Args:
    - model (torch.nn.Module): El modelo a entrenar.
    - dataloader (torch.utils.data.DataLoader): Dataloader para los datos de entrenamiento.
    - optimizer (torch.optim.Optimizer): Optimizador.
    - criterion (torch.nn.Module): Función de pérdida.
    - device (torch.device): Dispositivo para la ejecución (cuda o cpu).

    Retorna:
    - float: Pérdida promedio durante la época.
    """

    model.train()
    total_loss = 0.0

    for batch in dataloader:
        # Mueve los datos al dispositivo correspondiente
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        # Backward pass y optimización
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """
    Realiza una época de validación.

    Args:
    - model (torch.nn.Module): El modelo a evaluar.
    - dataloader (torch.utils.data.DataLoader): Dataloader para los datos de validación.
    - criterion (torch.nn.Module): Función de pérdida.
    - device (torch.device): Dispositivo para la ejecución (cuda o cpu).

    Retorna:
    - float: Pérdida promedio durante la época.
    """

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

    return total_loss / len(dataloader)
