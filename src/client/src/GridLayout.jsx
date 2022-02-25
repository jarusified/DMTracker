import React from 'react'
import { makeStyles } from '@material-ui/core/styles';
import { Grid } from '@mui/material';

import CommWrapper from './components/CommWrapper';
import CCTWrapper from './components/CCTWrapper';
import ReuseWrapper from './components/ReuseWrapper';
import MetricsWrapper from './components/MetricsWrapper';
import TimelineWrapper from './components/TimelineWrapper';

const useStyles = makeStyles((theme) => ({
    contentContainer: {
        height: '100%',
        width: window.innerWidth,
        flexGrow: 1,
        alignItems: 'stretch',
        flexWrap: 'wrap',
    },
    rowContainer: {
        display: "grid",
        gridAutoFlow: "column",
        alignItems: "center",
        padding: 0,
    }
}))

export default function GridLayout() {
    const classes = useStyles();
    return (
        <Grid className={classes.contentContainer}>
            <Grid container item direction='column'>
                <Grid className={classes.rowContainer}>
                    <CommWrapper />
                    <CCTWrapper />
                </Grid>
                <Grid className={classes.rowContainer}> 
                    <ReuseWrapper />
                    <MetricsWrapper />
                </Grid>
                <Grid className={classes.rowContainer}>
                    <TimelineWrapper />
                </Grid>
            </Grid>
        </Grid>
    )
}
