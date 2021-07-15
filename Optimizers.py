import numpy as np

def backward(a, y, optim, params):
	square = np.vectorize(lambda x: x*x)
	x = params["batch"]
	w = params["weights"]
	bias = params["bias"]

	Loss = (1/(2*x.shape[0])) * square((y - a))
	params["loss"].append(np.sum(Loss))

	dw = (y - a)
	db = sum(dw)/(-x.shape[0])
	dw = np.reshape(dw, (1, -1))
	dw = np.dot(dw, x)
	dw = np.reshape(dw, (-1, 1))
	dw/=(-x.shape[0])


	if optim == "GD":
		lr = params["lr"]
		w -= 0.001*dw
		bias -= db

	elif optim == "Momentum":
		params["w_update"] = params["gamma"]*params["w_update"] + params["lr"]*dw
		params["b_update"] = params["gamma"]*params["b_update"] + params["lr"]*db
		w -= params["w_update"]
		bias -= params["b_update"]

	elif optim == "Adagrad":
		params["w_decay"] += dw*dw
		params["b_decay"] += db*db
		w -= params["lr"] * (dw / (np.sqrt(params["w_decay"]) + 1e-7))
		bias -= params["lr"] * (db / (np.sqrt(params["b_decay"]) + 1e-7))

	elif optim == "RMSProp":
		params["w_decay"] += params["beta"]*params["w_decay"] + dw*dw
		params["b_decay"] += params["beta"]*params["b_decay"] + db*db
		w -= params["lr"] * (dw / (np.sqrt(params["w_decay"]) + 1e-7))
		bias -= params["lr"] * (db / (np.sqrt(params["b_decay"]) + 1e-7))

	elif optim == "Adam":
		params["w_grad"] = params["beta1"]*params["w_grad"] + (1-params["beta1"])*dw
		params["w_decay"] = params["beta2"]*params["w_decay"] + (1-params["beta2"])*(dw*dw)
		params["b_grad"] = params["beta1"]*params["b_grad"] + (1-params["beta1"])*db
		params["b_decay"] = params["beta2"]*params["b_decay"] + (1-params["beta2"])*(db*db)

		w_m = params["w_grad"] / (1 - params["beta1"]**params["t"])
		w_v = params["w_decay"] / (1 - params["beta2"]**params["t"])
		b_m = params["b_grad"] / (1 - params["beta1"]**params["t"])
		b_v = params["b_decay"] / (1 - params["beta2"]**params["t"])

		w -= params["lr"] * (w_m / (np.sqrt(w_v) + 1e-7))
		bias -= params["lr"] * (b_m / (np.sqrt(b_v) + 1e-7))

	params["weights"] = w
	params["bias"] = bias
	return params


def predict(X, params):
	return np.dot(X, params["weights"]) + params["bias"]


def fit(X, y, optim, params, epochs):
	params["batch"] = X
	params["loss"] = list()
	params["weights"] = np.random.rand(X.shape[1], 1)*0.001
	params["bias"] = 1
	y = np.reshape(y, (-1, 1))

	for t in range(1, epochs+1):
		w = params["weights"]
		bias = params["bias"]
		params["t"] = t

		a = np.dot(X, w) + bias
		params = backward(a, y, optim, params)
