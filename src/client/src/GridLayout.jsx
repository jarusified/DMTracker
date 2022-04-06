import React from 'react'
import { makeStyles } from '@material-ui/core/styles';
import { Grid } from '@mui/material';

import CommWrapper from './components/CommWrapper';
import CCTWrapper from './components/CCTWrapper';
import ReuseWrapper from './components/ReuseWrapper';
import MetricsWrapper from './components/MetricsWrapper';
import TimelineWrapper from './components/TimelineWrapper';
import KernelWrapper from './components/KernelWrapper';

const useStyles = makeStyles((theme) => ({
    contentContainer: {
        height: '100%',
        width: window.innerWidth,
        paddingTop: 60,
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
        <Grid className={classes.contentContainer} justify="center">
            <Grid container item>
                <Grid className={classes.rowContainer} xs={12} item={true}> 
                    <KernelWrapper />
                    <MetricsWrapper />
                    <CCTWrapper />
                </Grid>
                <Grid className={classes.rowContainer} xs={12} item={true}>
                    <CommWrapper />
                </Grid>
                <Grid className={classes.rowContainer} xs={12} item={true}>
                    <TimelineWrapper />
                </Grid>
            </Grid>
        </Grid>
    )
}
