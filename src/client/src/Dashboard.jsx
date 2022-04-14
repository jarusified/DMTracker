import React from "react";
import {
	Box,
	CssBaseline,
} from "@mui/material";

import DashboardLayout from "./DashboardLayout";
import ControlWrapper from "./components/ControlWrapper";
import ToolBar from "./components/ToolBar";

export default function Dashboard() {
	return (
		<Box
			sx={{
				display: "flex",
				boxShadow: 1,
				width: "inherit",
			}}
		>
			<CssBaseline />
			<ToolBar />
			<ControlWrapper />
			{/* <DashboardLayout /> */}
		</Box>
	);
}
