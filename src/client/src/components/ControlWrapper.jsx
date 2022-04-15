import React, { Fragment, useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import { styled, useTheme } from "@mui/material/styles";
import { makeStyles } from "@material-ui/core/styles";
import { 
    IconButton,
    Typography,
    Divider,
    List,
    ListItem,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    Button,
} from "@mui/material";

import MuiDrawer from "@mui/material/Drawer";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";

import { updateSelectedMetric, updateSelectedKernel, fetchMetrics, fetchKernels } from "../actions";

const DRAWER_WIDTH = 240;

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

const useStyles = makeStyles(() => ({
	formControl: {
		padding: 20,
		justifyContent: "flex-end",
		textColor: "white",
	},
}));


export default function ControlWrapper() {
    const classes = useStyles();
	const theme = useTheme();
	const dispatch = useDispatch();

	const [open, setOpen] = React.useState(false);

	const handleDrawerOpen = () => {
		setOpen(true);
	};

	const handleDrawerClose = () => {
		setOpen(false);
	};

    const experiments = useSelector((store) => store.experiments);
    const selectedExperiment = useSelector((store) => store.selected_experiment);

    const metrics = useSelector((store) => store.metrics);
	const selectedMetric = useSelector((store) => store.selected_metric);

	const kernels = useSelector((store) => store.kernels);
	const selectedKernel = useSelector((store) => store.selected_kernel);

    useEffect(() => {
		if (metrics.length == 0) {
			dispatch(fetchMetrics());
		}
	}, []);

	useEffect(() => {
		if (kernels.length == 0) {
			dispatch(fetchKernels());
		}
	}, [kernels]);

	return (
		<Fragment>
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
									</MenuItem>
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
		</Fragment>
	);
}
