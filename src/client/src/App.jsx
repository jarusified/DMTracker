import React from "react";
import { MuiThemeProvider, createTheme } from "@material-ui/core/styles";
import { grey } from "@material-ui/core/colors";

import { Provider } from "react-redux";
import { createStore, applyMiddleware } from "redux";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import thunkMiddleware from "redux-thunk";

import Reducer from "./reducer";
import "./main.css";
import DashboardLayoutWrapper from "./DashboardLayoutWrapper";

const store = createStore(Reducer, applyMiddleware(thunkMiddleware));

// Adjust the color theme here.
const theme = createTheme({
	palette: {
		primary: {
			main: grey[700],
		},
		secondary: {
			main: grey[700],
		},
	},
});

function App() {
	return (
		<Provider store={store}>
			<Router>
				<Routes>
					<Route path="/" element={<DashboardWrapper />} />
				</Routes>
			</Router>
		</Provider>
	);
}

function DashboardWrapper() {
	return (
		<MuiThemeProvider theme={theme}>
			<DashboardLayoutWrapper />
		</MuiThemeProvider>
	);
}

export default App;
