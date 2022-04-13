import React, { useEffect } from "react";
import { useSelector, useDispatch } from "react-redux";
import { 
    Toolbar,
    IconButton,
    Typography,
    FormControl,
    Select,
    InputLabel,
    FormHelperText,
    MenuItem
} from "@mui/material";

import { styled } from "@mui/material/styles";
import { makeStyles } from "@material-ui/core/styles";
import MuiAppBar from "@mui/material/AppBar";
import MenuIcon from "@mui/icons-material/Menu";
import { fetchExperiments } from "../actions";

const DRAWER_WIDTH = 240;

const useStyles = makeStyles((theme) => ({
	toolbar: {
		color: "black",
		backgroundColor: "white",
		justifyContent: "space-between",
	},
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

const handleDrawerOpen = () => {
    setOpen(true);
};

const handleDrawerClose = () => {
    setOpen(false);
};


export default function ToolBar() {
    const classes = useStyles();
    const dispatch = useDispatch();
    const experiments = useSelector((store) => store.experiments);
    const selectedExperiment = useSelector((store) => store.selectedExperiment);

    useEffect(() => {
        dispatch(fetchExperiments());
	}, []);

    return (
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
    )
}