import React, { useEffect } from "react";
import { MuiThemeProvider, createTheme } from "@material-ui/core/styles";
import { grey } from "@material-ui/core/colors";

import { Provider } from "react-redux";
import { createStore, applyMiddleware } from "redux";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import thunkMiddleware from "redux-thunk";

import Reducer from "./reducer";
import "./main.css";
import Dashboard from "./Dashboard";
import Summary from "./Summary";

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
		background: {
			paper: "#fff",
		},
		text: {
			primary: {
				main: '#fff',
			},
			secondary: {
				main: '#000',
			}
		}
	},
});

function App() {
	useEffect(() => {
		document.title = "DataFlow - Data Movement in CPU-GPU Interfaces";
	}, []);
	return (
		<Provider store={store}>
			<Router>
				<Routes>
					<Route path="/" element={<SummaryWrapper />} />
					<Route path="/dashboard" element={<DashboardWrapper />} />
				</Routes>
			</Router>
		</Provider>
	);
}

function DashboardWrapper() {
	return (
		<MuiThemeProvider theme={theme}>
			<Dashboard />
		</MuiThemeProvider>
	);
}

function SummaryWrapper() {
	return (
		<MuiThemeProvider theme={theme}>
			<Summary />
		</MuiThemeProvider>
	);
}

export default App;
