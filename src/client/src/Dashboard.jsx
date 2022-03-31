import React, { useState, useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import { styled, useTheme } from "@mui/material/styles";
import { makeStyles } from "@material-ui/core/styles";
import MuiDrawer from "@mui/material/Drawer";
import MuiAppBar from "@mui/material/AppBar";
import {
	Box,
	Toolbar,
	List,
	CssBaseline,
	Typography,
	Divider,
	IconButton,
	ListItem,
	ListItemText,
	FormControl,
	InputLabel,
	FormHelperText,
	MenuItem,
	Select,
} from "@mui/material";
import MenuIcon from "@mui/icons-material/Menu";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";

import GridLayout from "./GridLayout";
import { fetchExperiments, updateSelectedExperiment, updateSelectedMetric, updateSelectedKernel, fetchMetrics, fetchKernels } from "./actions";

const DRAWER_WIDTH = 240;

const useStyles = makeStyles((theme) => ({
	toolbar: {
		color: "black",
		backgroundColor: "white",
		justifyContent: "space-between",
	},
	formControl: {
		padding: 20,
		justifyContent: "flex-end",
		textColor: "white",
	},
}));

const openedMixin = (theme) => ({
	width: DRAWER_WIDTH,
	transition: theme.transitions.create("width", {
		easing: theme.transitions.easing.sharp,
		duration: theme.transitions.duration.enteringScreen,
	}),
	overflowX: "hidden",
});

const closedMixin = (theme) => ({
	transition: theme.transitions.create("width", {
		easing: theme.transitions.easing.sharp,
		duration: theme.transitions.duration.leavingScreen,
	}),
	overflowX: "hidden",
	width: `calc(${theme.spacing(0)} + 1px)`,
	[theme.breakpoints.up("sm")]: {
		width: `calc(${theme.spacing(0)} + 1px)`,
	},
});

const DrawerHeader = styled("div")(({ theme }) => ({
	display: "flex",
	alignItems: "center",
	justifyContent: "flex-end",
	// necessary for content to be below app bar
	...theme.mixins.toolbar,
}));

const AppBar = styled(MuiAppBar, {
	shouldForwardProp: (prop) => prop !== "open",
})(({ theme, open }) => ({
	zIndex: theme.zIndex.drawer + 1,
	transition: theme.transitions.create(["width", "margin"], {
		easing: theme.transitions.easing.sharp,
		duration: theme.transitions.duration.leavingScreen,
	}),
	...(open && {
		marginLeft: DRAWER_WIDTH,
		width: `calc(100% - ${DRAWER_WIDTH}px)`,
		transition: theme.transitions.create(["width", "margin"], {
			easing: theme.transitions.easing.sharp,
			duration: theme.transitions.duration.enteringScreen,
		}),
	}),
}));

const Drawer = styled(MuiDrawer, {
	shouldForwardProp: (prop) => prop !== "open",
})(({ theme, open }) => ({
	width: DRAWER_WIDTH,
	flexShrink: 0,
	whiteSpace: "nowrap",
	boxSizing: "border-box",
	...(open && {
		...openedMixin(theme),
		"& .MuiDrawer-paper": openedMixin(theme),
	}),
	...(!open && {
		...closedMixin(theme),
		"& .MuiDrawer-paper": closedMixin(theme),
	}),
}));

export default function Dashboard() {
	const classes = useStyles();

	const dispatch = useDispatch();
	const experiments = useSelector((store) => store.experiments);
	const selectedExperiment = useSelector((store) => store.selected_experiment);

	const metrics = useSelector((store) => store.metrics);
	const selectedMetric = useSelector((store) => store.selected_metric);

	const kernels = useSelector((store) => store.kernels);
	const selectedKernel = useSelector((store) => store.selected_kernel);

	useEffect(() => {
		if (experiments.length == 0) {
			dispatch(fetchExperiments());
		}

		if (metrics.length == 0) {
			dispatch(fetchMetrics());
		}
	}, []);

	useEffect(() => {
		if (kernels.length == 0 && selectedExperiment !== "" && experiments.length > 0) {
			dispatch(fetchKernels(selectedExperiment));
		}
	}, [experiments, selectedExperiment]);

	const theme = useTheme();
	const [open, setOpen] = React.useState(false);

	const handleDrawerOpen = () => {
		setOpen(true);
	};

	const handleDrawerClose = () => {
		setOpen(false);
	};

	return (
		<Box
			sx={{
				display: "flex",
				boxShadow: 1,
				width: "inherit",
			}}
		>
			<CssBaseline />
			<AppBar position="fixed" open={open}>
				<Toolbar className={classes.toolbar}>
					<IconButton
						color="inherit"
						aria-label="open drawer"
						onClick={handleDrawerOpen}
						edge="start"
						sx={{
							marginRight: "36px",
							...(open && { display: "none" }),
						}}
					>
						<MenuIcon />
					</IconButton>
					<Typography variant="h6" noWrap component="div">
						DataFlow - Analysis of CPU-GPU Data Movement
					</Typography>
					<Typography variant="text" noWrap component="div">
						Ensemble: {experiments.length} runs
					</Typography>
					{experiments.length > 0 ? (
						<FormControl className={classes.formControl} size="small">
							<InputLabel id="dataset-label">Experiments</InputLabel>
							<Select
								labelId="dataset-label"
								id="dataset-select"
								value={selectedExperiment}
								onChange={(e) => {
									dispatch(updateSelectedExperiment(e.target.value));
								}}
							>
								{experiments.map((cc) => (
									<MenuItem key={cc} value={cc}>
										{cc}
									</MenuItem>
								))}
							</Select>
							<FormHelperText>Select the experiment</FormHelperText>
						</FormControl>
					) : (
						<></>
					)}
				</Toolbar>
			</AppBar>
			<Drawer variant="permanent" open={open}>
				<DrawerHeader>
					<IconButton onClick={handleDrawerClose}>
						{theme.direction === "rtl" ? (
							<ChevronRightIcon />
						) : (
							<ChevronLeftIcon />
						)}
					</IconButton>
				</DrawerHeader>
				<Divider />
				<List>
					<Typography variant="text" noWrap component="div">
						Ensemble Performance
					</Typography>
					<ListItem button key={"Select Kernel"}>
						<FormControl className="form-control-fit">
							<InputLabel className={classes.formLabel} id="kernel-label">
								Select the Kernel
							</InputLabel>
							<Select
								labelId="kernel-label"
								id="kernel-select"
								value={selectedKernel}
								onChange={(e) => {
									dispatch(updateSelectedKernel(e.target.value));
								}}
							>
								{kernels.map((cc) => (
									<MenuItem key={cc} value={cc}>
										{cc}
									</MenuItem	>
								))}
							</Select>
						</FormControl>
					</ListItem>
					<ListItem button key={"Select Metric"}>
						<FormControl className="form-control-fit">
							<InputLabel className={classes.formLabel} id="metric-label">
								Select the Metric
							</InputLabel>
							<Select
								labelId="metric-label"
								id="metric-select"
								value={selectedMetric}
								onChange={(e) => {
									dispatch(updateSelectedMetric(e.target.value));
								}}
							>
								{metrics.map((cc) => (
									<MenuItem key={cc} value={cc}>
										{cc}
									</MenuItem>
								))}
							</Select>
						</FormControl>
					</ListItem>
				</List>
				<Divider />
			</Drawer>
			<DrawerHeader />
			<GridLayout />
		</Box>
	);
}
