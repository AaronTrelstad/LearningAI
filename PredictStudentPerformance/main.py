import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Read CSV data into Pandas dataframe
dataframe = pd.read_csv('studentData.csv')

# Replace all missing data with 0
dataframe.fillna(0, inplace=True)

# Selects columns with the correct datatypes
dataframe = dataframe.select_dtypes(include=[float, int])

# Remove the student "StudentID" column
dataframe.drop(['StudentID'], axis=1, inplace=True)

# Remove the "GPA" and "GradeClass" from the set of features
features = dataframe.drop(["GPA", "GradeClass"], axis=1).values

# Create the target of "GPA"
target = dataframe["GPA"].values

# Initialize a scaler to normalize the data
scaler = StandardScaler()

# Normalize the features so they have a mean of 0 and standard deviation of 1
features = scaler.fit_transform(features)

# Create a tensor for the feeatures
features = torch.tensor(features, dtype=torch.float32)

# Create a tensor for the target and reshape
target = torch.tensor(target, dtype=torch.float32).unsqueeze(1)

# Split the data into training and test sets, (80% training, 20% testing)
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Creates a dataset by merging the features with the targets
train_dataset = TensorDataset(features_train, target_train)
test_dataset = TensorDataset(features_test, target_test)

# Splits the dataset into smaller batchs, shuffling prevents the model from memorizing the order
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Create a basic Neural Network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()

        # First layer maps the number of input features to 10 output features
        self.firstConnectedLayer = nn.Linear(features_train.shape[1], 10)

        # Second layer combines the 10 outputs fromt he first layer into 1 value
        self.secondConnectedLayer = nn.Linear(10, 1)
    
    def forward(self, data):
        # Use a RELU (Rectified Linear Unit) activation function on the first layer to produce 10 outputs
        result = torch.relu(self.firstConnectedLayer(data))

        # Combine the 10 outputs into a singular value using the second layer
        result = self.secondConnectedLayer(result)

        return result
    
model = SimpleNN()

# Create variable to hold the Mean Squared Error so that we can track preformance
error = nn.MSELoss()

# I use Adam Optimization which uses Stochastic Gradient Descent and I initialize a learning rate of 0.001
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of training rounds
num_epochs = 100

for epoch in range(num_epochs):
    # Tells the model that it's time to learn
    model.train()

    # Gives the model a sample of students (feature_batch) and GPA (target_batch) to learn from
    for feature_batch, target_batch in train_loader:
        # Resets the gradients
        optimizer.zero_grad()

        # Runs the model of the students and collects their expected GPA
        outputs = model(feature_batch)

        # Compares the expected GPA (outputs) to the actual (target_batch)
        loss = error(outputs, target_batch)

        # Computes the gradient of the current tensor
        loss.backward()

        # Proforms a update based on the current gradient
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# Tell the model we are testing
model.eval()

# We don't want the model to learn we just want to test
with torch.no_grad():
    # Track the mistakes
    test_loss = 0

    # Gives the model a sample of students (feature_batch) and GPA (test_batch) to test on
    for feature_batch, target_batch in test_loader:
        # Runs the model of the students and collects their expected GPA
        outputs = model(feature_batch)
        
        # Compares the expected GPA (outputs) to the actual (target_batch)
        loss = error(outputs, target_batch)

        # Adds the loss to the total loss
        test_loss += loss.item()
    print(f'Test Loss: {test_loss / len(test_loader):.4f}')

# Function to predict the GPA of a given student
def predict_gpa(student_stats):
    # Transform the students data using the previously computed statistics
    student_stats = scaler.transform([student_stats])

    # Converts the data into a tensor
    student_stats = torch.tensor(student_stats, dtype=torch.float32)

    # Tell the model we are testing
    model.eval()

    # We don't want the model to learn we just want to test
    with torch.no_grad():
        # Create the GPA prediction
        gpa_predication = model(student_stats)
    return gpa_predication.item()

# Sample student
sample_student_dict = {
    "Age": 16,
    "Gender": 0,
    "Ethnicity": 1,
    "ParentalEducation": 4,
    "StudyTimeWeekly": 10,
    "Absences": 0,
    "Tutoring": 1,
    "ParentalSupport": 3,
    "Extracurricular": 1,
    "Sports": 1,
    "Music": 1,
    "Volunteering": 1,
}

sample_student_list = [
    sample_student_dict["Age"],
    sample_student_dict["Gender"],
    sample_student_dict["Ethnicity"],
    sample_student_dict["ParentalEducation"],
    sample_student_dict["StudyTimeWeekly"],
    sample_student_dict["Absences"],
    sample_student_dict["Tutoring"],
    sample_student_dict["ParentalSupport"],
    sample_student_dict["Extracurricular"],
    sample_student_dict["Sports"],
    sample_student_dict["Music"],
    sample_student_dict["Volunteering"],
]

# Calculates the expected GPA of the sample student
predicted_gpa = predict_gpa(sample_student_list)
print(f'Predicted GPA: {predicted_gpa:.4f}')
