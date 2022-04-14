import React, { useEffect, useState } from "react";
import { useSelector, useDispatch } from "react-redux";
import {
	Toolbar,
	FormHelperText,
	IconButton,
    Typography,
    Divider,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
} from "@mui/material";

import { styled } from "@mui/material/styles";
import { makeStyles } from "@material-ui/core/styles";
import MuiAppBar from "@mui/material/AppBar";
import MenuIcon from "@mui/icons-material/Menu";
import MuiDrawer from "@mui/material/Drawer";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";

import { fetchExperiments } from "../actions";
import { useTheme } from "@emotion/react";

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

export default function ToolBar() {
    const theme = useTheme();
	const classes = useStyles();
	const dispatch = useDispatch();
	const experiments = useSelector((store) => store.experiments);
	const selectedExperiment = useSelector((store) => store.selected_experiment);
	const [open, setOpen] = useState(false);

	const handleDrawerOpen = () => {
		setOpen(true);
	};

	const handleDrawerClose = () => {
		setOpen(false);
	};
	useEffect(() => {
		dispatch(fetchExperiments());
	}, []);

	return (
        <>
            <AppBar position="fixed" open={open} elevation={1}>
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
            </Drawer>
        </>

	);
}
