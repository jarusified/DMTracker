import React, { Fragment } from 'react'
import {
    Grid, Paper, Typography,
} from '@material-ui/core';


function TimelineWrapper() {
    return (
        <Fragment>  
            <Paper elevation={3}>
                <Typography variant='overline' style={{ fontWeight: 'bold' }}>
                    Execution Timeline
                </Typography>
                <Grid container item direction='column' spacing={2}>
                    <Timeline />
                </Grid>
                <Grid container item direction='column' spacing={2}>
                </Grid>
            </Paper>
        </Fragment>
    )
}

function Timeline() {
    return (
        <Fragment>
            <svg width={window.innerWidth} height={window.innerHeight/4}></svg>
        </Fragment>
    )
}


export default TimelineWrapper;
