# hackathon
Hackathon AV - Approach

The dataset had 9 Features associated with the Target Is_Lead, but correlation associated was not so strong.
So after a series of exploratory analysis, I decided to transform features to enhance associativity, which helped improve the model performance.

Feature Engineering:

Region_Code - Transormed with associated probabilities of being a lead in that region.

AvgAcountBalance - Categorical transformation to classes of Incomes.

Preprocessing:

Numerical Columns were scaled using MinMaxScaler() to acheive better performance.

Categorical Columns were changed to labels using OneHotEncoder().

Models: 
1. RandomForestClassifier
2. VotingClassifier - (RandomForestClassifier, LogisticRegression)
3. NeuralNetworks - Keras.Sequential() with 6 Hidden Layers, activation = ReLU, Sigmoid.

Results : Validation Score : 85.7 %, Test Score - 73.7%
