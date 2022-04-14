import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import { Grid } from '@material-ui/core';
import {
    Box,
    Paper,  
    CssBaseline
} from "@mui/material";

import ToolBar from "./components/ToolBar";
import KernelWrapper from './components/KernelWrapper';
import MetricsWrapper from './components/MetricsWrapper';

const useStyles = makeStyles((theme) => ({
    root: {
        display: "flex",
        flexGrow: 1,
    },
    paper: {
        textAlign: 'center',
        color: theme.palette.text.secondary,
    }
}))

export default function SummaryWrapper() {
    const classes = useStyles();

    return (
        <div className={classes.root}>
            <CssBaseline />
            <ToolBar />
            <Box
                sx={{
                    display: "flex",
                    position: "relative",
                    width: "100%",
                    top: "0px",
                }}
            >
                <Grid container>
                    <Grid item xs={6}>
                        <Paper className={classes.paper}>
                            <KernelWrapper />
                        </Paper>
                    </Grid>
                    <Grid item xs={6}>
                        <Paper className={classes.paper}>
                            <MetricsWrapper />
                        </Paper>
                    </Grid>
                </Grid>
            </Box>
        </div>
    )
}
