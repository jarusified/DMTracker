import React, { Fragment } from 'react'
import {
    Grid, Paper, Typography,
} from '@material-ui/core';


function MetricsWrapper() {
    return (
        <Fragment>  
            <Paper elevation={3}>
                <Typography variant='overline' style={{ fontWeight: 'bold' }}>
                    Performance metrics
                </Typography>
                <Grid container item direction='column' spacing={2}>
                    <Metrics />
                </Grid>
                <Grid container item direction='column' spacing={2}>
                </Grid>
            </Paper>
        </Fragment>
    )
}

function Metrics() {
    return (
        <Fragment>
            <svg width={window.innerWidth/2} height={window.innerHeight/4}></svg>
        </Fragment>
    )
}


export default MetricsWrapper;
