import numpy as np
import streamlit as st
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Optimizers import fit, backward, predict


def load_data(dataset_name):
		if dataset_name == "Housing":
			data = datasets.load_boston()
		else:
			data = datasets.load_diabetes()
		X = data.data
		y = data.target
		return X, y


def get_params(optim):
		params = dict()
		params["lr"] = st.sidebar.slider("Learning Rate", min_value=0.01, max_value=1.0, step=0.01)

		if optim == "Momentum":
			params["gamma"] = st.sidebar.slider("Gamma", min_value=0.0, max_value=1.0, step=0.05)
			params["w_update"] = 0
			params["b_update"] = 0
		elif optim == "Adagrad":
			params["w_decay"] = 0
			params["b_decay"] = 0
		elif optim == "RMSProp":
			params["beta"] = st.sidebar.slider("Beta", min_value=0.0, max_value=1.0, step=0.05)
			params["w_decay"] = 0
			params["b_decay"] = 0
		elif optim == "Adam":
			params["beta1"] = st.sidebar.slider("Beta1", min_value=0.0, max_value=1.0, step=0.05)
			params["beta2"] = st.sidebar.slider("Beta2", min_value=0.0, max_value=1.0, step=0.05)
			params["w_decay"] = 0
			params["b_decay"] = 0
			params["w_grad"] = 0
			params["b_grad"] = 0
		return params


def main():
	st.sidebar.write("# Neural Network Optimizers")

	dataset_name = st.sidebar.selectbox("Select Dataset", ("Housing", "Diabetes"))
	optim = st.sidebar.selectbox("Select Optimizer", ("GD", "Momentum", "RMSProp", "Adagrad", "Adam"))
	epochs = st.sidebar.slider("Epochs", 1, 500, 10)


	X, y = load_data(dataset_name)
	scaler = StandardScaler()
	X = scaler.fit_transform(X)
	maximum = max(y)
	y = y/maximum

	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

	params = get_params(optim)
	fit(x_train, y_train, optim, params, epochs)
	y_pred = predict(x_test, params)

	y_pred = y_pred*maximum
	y_test = y_test*maximum

	square = np.vectorize(lambda x: x*x)
	loss = (1/(2*x_test.shape[0])) * square((y_test - y_pred))
	loss = np.sum(loss)
	st.write(f"### {optim} Loss: {loss}")

	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,11))
	ax1.plot(params["loss"])
	ax2.plot(y_test[:100], label='Truth')
	ax2.plot(y_pred[:100], label='Predicted')

	ax1.title.set_text("Training loss")
	ax2.title.set_text("Predictions")
	ax2.legend()
	st.pyplot(fig)


if __name__ == '__main__':
	main()
