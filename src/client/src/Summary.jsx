import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import { Grid } from '@material-ui/core';
import {
    Box,
    CssBaseline
} from "@mui/material";

import ToolBar from "./components/ToolBar";
import KernelWrapper from './components/KernelWrapper';
import MetricsWrapper from './components/MetricsWrapper';

const useStyles = makeStyles((theme) => ({
    contentContainer: {
        padding: theme.spacing(1),
        height: '100%',
        flexGrow: 1,
        alignItems: 'stretch',
        flexWrap: 'nowrap',
    },
    rowContainer: {
        display: "grid",
        gridAutoFlow: "column",
        alignItems: "center",
        justifyContent: "space-evenly",
        gridAutoColumns: "1fr",
    }
}))

export default function SummaryWrapper() {
    const classes = useStyles();

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
            <Grid className={classes.contentContainer}>
                <Grid container>
                    <Grid className={classes.rowContainer} item xs={12}>
                        <KernelWrapper />
                    </Grid>
                </Grid>
                <Grid container>
                    <Grid className={classes.rowContainer} item xs={12}>
                        <MetricsWrapper />
                    </Grid>
                </Grid>
            </Grid>
        </Box>
    )
}
