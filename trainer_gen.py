import torch
import torch.nn as nn
import torch.optim as optim
import sys
def get_simmat():
    pass


def train_model(train_loader, test_loader, model, criterion=nn.CrossEntropyLoss(), optimizer=optim.SGD, lr=0.001, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optimizer(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_predictions / total_predictions * 100

        # Evaluation
        model.eval()
        test_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                test_loss += loss.item()

        test_loss /= len(test_loader)
        test_accuracy = correct_predictions / total_predictions * 100

        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f} Train Accuracy: {train_accuracy:.2f}% - "
              f"Test Loss: {test_loss:.4f} Test Accuracy: {test_accuracy:.2f}%")
