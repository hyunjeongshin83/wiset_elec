import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

def create_model(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    start_region = pd.read_csv("start_region.csv")
    critical_region = pd.read_csv("critical_region.csv")

    X_start = start_region[['time', 'usage_time', 'voltage']].values
    y_start = start_region['device'].values
    X_critical = critical_region[['time', 'usage_time', 'voltage']].values
    y_critical = critical_region['device'].values

    model_start = create_model(X_start.shape[1])
    model_start.fit(X_start, y_start, epochs=50, batch_size=10, verbose=0)
    model_start.save('model_start.h5')

    model_critical = create_model(X_critical.shape[1])
    model_critical.fit(X_critical, y_critical, epochs=50, batch_size=10, verbose=0)
    model_critical.save('model_critical.h5')

    print('Model training complete.')
